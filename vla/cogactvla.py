"""
cogactvla.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from copy import deepcopy
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
import torch.nn.functional as F
from transformers import LlamaTokenizerFast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.materialize import get_vision_backbone_and_transform
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
from prismatic.util.cot_utils import CotTag, get_cot_tags_list

from action_model.action_model import ActionModel
from action_model.models import DiT

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class MemoryTransformerBlock(nn.Module):
    def __init__(self, feature_dim: int):
        """
        :param feature_dim: 特征维度，例如 4096
        """
        super().__init__()
        self.feature_dim = feature_dim

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # query: (B, 1, D), memory: (B, N, D)
        # --- Self-Attention 部分 ---
        q_self = self.q_proj(query)
        self_attn = F.scaled_dot_product_attention(
            q_self, q_self, q_self, dropout_p=0.0, is_causal=False)
        query = self.norm1(query + self_attn)

        # --- Cross-Attention 部分 ---
        # 此处直接使用更新后的 query 与 memory，不做额外线性变换
        cross_attn = F.scaled_dot_product_attention(
            query, memory, memory, dropout_p=0.0, is_causal=False)
        query = self.norm2(query + cross_attn)

        # --- FFN ---
        ffn_out = self.ffn(query)
        query = self.norm3(query + ffn_out)
        return query


class MemoryAttention(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int = 2):
        """
        :param feature_dim: 特征维度
        :param num_layers: 堆叠层数，默认 2 层
        :param dropout: dropout 概率
        """
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryTransformerBlock(feature_dim)
            for _ in range(num_layers)
        ])

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, memory)
        return query


class MemoryBank(nn.Module):
    def __init__(self, feature_dim: int, traj_group_size: int, max_memory_size: int):
        """
        :param feature_dim: 特征维度，例如 4096
        :param traj_group_size: 训练时每个 episode 内的样本数；测试时 bs=1 每次 forward 属于同一 episode
        :param max_memory_size: 测试时允许保留的最大 memory 数，例如 50，当超过时进行均匀重采样
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.traj_group_size = traj_group_size
        self.max_memory_size = max_memory_size
        self.memory_attention = MemoryAttention(feature_dim, num_layers=2)
        self.reset()

    def reset(self):
        """
        重置 memory bank 与帧计数器
        测试时统一认为只有一个 episode（episode_id=0）
        """
        self.bank = {
        }  # 存储格式： {episode_id: [(global_frame_idx, feature), ...]}
        self.global_frame_counter = {}  # 每个 episode 内的全局帧号

    def update_memory(self, batch_idx: int, feature: torch.Tensor):
        """
        更新 memory：
          - 训练时：按照 batch_idx // traj_group_size 来区分 episode
          - 测试时：统一用 episode_id=0，并使用 global_frame_counter 记录全局帧顺序。
          如果测试时存储数超过 max_memory_size，则执行均匀重采样。
        """
        if self.training:
            episode_id = batch_idx // self.traj_group_size
        else:
            episode_id = 0

        if episode_id not in self.bank:
            self.bank[episode_id] = []
            self.global_frame_counter[episode_id] = 0

        curr_frame = self.global_frame_counter[episode_id]
        self.bank[episode_id].append((curr_frame, feature.detach().clone()))
        self.global_frame_counter[episode_id] = curr_frame + 1

        if not self.training:
            if len(self.bank[episode_id]) > self.max_memory_size:
                # FIFO方式：直接丢弃最早的帧，直到长度等于max_memory_size
                while len(self.bank[episode_id]) > self.max_memory_size:
                    self.bank[episode_id].pop(0)

    def attend_memory(self, batch_idx: int, feature: torch.Tensor) -> torch.Tensor:
        """
        计算当前帧与对应 episode 内历史帧的 memory attention 融合。
        测试时不依赖局部 batch_idx，而是使用 global_frame_counter 判断当前全局帧号。
        """
        if self.training:
            episode_id = batch_idx // self.traj_group_size
            sample_order = batch_idx % self.traj_group_size
        else:
            episode_id = 0
            sample_order = self.global_frame_counter.get(0, 0)  # 当前待写入帧号即为已存帧数

        if sample_order == 0 or (episode_id not in self.bank or len(self.bank[episode_id]) == 0):
            return feature
        memory_features = [item[1] for item in self.bank[episode_id]]
        total = len(memory_features)
        if total > self.traj_group_size:
            # 均匀选取traj_group_size个样本
            indices = torch.linspace(
                0, total - 1, steps=self.traj_group_size).floor().long().tolist()
            selected = [memory_features[i] for i in indices]
        else:
            selected = memory_features

        mem_tensor = torch.cat(selected, dim=0).unsqueeze(0)  # (1, Nx, D)
        query = feature.unsqueeze(0)  # (1, x, D)
        attended_feature = self.memory_attention(query, mem_tensor)
        return attended_feature.squeeze(0)

    def process_batch(self, cognition_features: torch.Tensor) -> torch.Tensor:
        """
        对一个 batch 的 cognition_features 依次进行 memory attention，并更新 memory bank。
        测试时（bs=1）每次 forward 只有一帧，但 global_frame_counter 会累积。
        """
        B = cognition_features.size(0)
        output_features = []
        for i in range(B):
            # 测试时忽略局部 i，直接使用当前 global_counter值
            idx = i if self.training else self.global_frame_counter.get(0, 0)
            feat = cognition_features[i]
            attended_feat = self.attend_memory(idx, feat)
            self.update_memory(idx, feat)
            output_features.append(attended_feat.unsqueeze(0))
        return torch.cat(output_features, dim=0)


class CrossTransformerBlock(nn.Module):
    """
    Attention + 残差 + LayerNorm + Feed‑Forward + LayerNorm
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)
        # Feed‑Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # query: (B, T, D), memory: (B, M, D)
        q = self.q_proj(query)
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False)
        # 残差 + LN
        x = self.attn_norm(query + attn_out)
        # FFN + LN
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)


class MemoryBankV2(nn.Module):
    """
    流程：
      1. query = cognition_features[i]               # (token_num, D)
      2. memory = concat(bank[episode_id])           # (M*token_num, D)
      3. attn = CrossBlocks(query, memory)          # (token_num, D)
      4. delta = attn - query
      5. gate  = sigmoid(gate_mlp([query, delta]))  # (token_num, D)
      6. fused = query + gate * delta
      7. update_memory(episode_id, original query)
    """

    def __init__(self, feature_dim: int, traj_group_size: int, max_memory_size: int, num_layers: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.traj_group_size = traj_group_size
        self.max_memory_size = max_memory_size
        # 多层 Cross‑Attention + FFN
        self.cross_blocks = nn.ModuleList([
            CrossTransformerBlock(feature_dim)
            for _ in range(num_layers)
        ])

        # Scale‑MLP：zero‑init weight, bias = +1 ⇒ scale≈1 ⇒ fused = query
        self.scale_mlp = nn.Linear(feature_dim * 2, feature_dim)
        nn.init.zeros_(self.scale_mlp.weight)
        nn.init.constant_(self.scale_mlp.bias, 1.0)

        self.reset()

    def reset(self):
        self.bank = {}
        self.global_frame_counter = {}

    def update_memory(self, batch_idx: int, feature: torch.Tensor):
        if self.training:
            episode_id = batch_idx // self.traj_group_size
        else:
            episode_id = 0
        if episode_id not in self.bank:
            self.bank[episode_id] = []
            self.global_frame_counter[episode_id] = 0
        fid = self.global_frame_counter[episode_id]
        self.bank[episode_id].append((fid, feature.detach().clone()))
        self.global_frame_counter[episode_id] = fid + 1
        if not self.training and len(self.bank[episode_id]) > self.max_memory_size:
            self.bank[episode_id] = self.bank[episode_id][-self.max_memory_size:]

    def process_batch(self, cognition_features: torch.Tensor) -> torch.Tensor:
        B, token_num, D = cognition_features.shape
        outputs = []
        for i in range(B):
            if self.training:
                episode_id = i // self.traj_group_size
            else:
                episode_id = 0
            query = cognition_features[i].unsqueeze(0)  # (1, token_num, D)
            hist = [feat for _, feat in self.bank.get(episode_id, [])]
            if hist:
                memory = torch.cat(hist, dim=0).unsqueeze(0)  # (1, M, D)
                x = query
                for block in self.cross_blocks:
                    x = block(x, memory)                         # cross + ffn
                attn_out = x
            else:
                attn_out = query
            # Gate 融合
            cat_qo = torch.cat([query, attn_out], dim=-1)  # (1, T, 2D)
            scale = torch.sigmoid(self.scale_mlp(cat_qo))  # 初始全 1
            fused = scale * query + (1 - scale) * attn_out  # 初始 fused == query
            outputs.append(fused.squeeze(0).unsqueeze(0))
            # 更新
            self.update_memory(i, cognition_features[i])
        return torch.cat(outputs, dim=0)  # (B, token_num, D)


class ResamplerBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=8, dropout=0.0, ff_mult=4):
        """
        融合跨注意力与 FFN 的模块块。
        参数：
          q_dim: latent (查询) 的维度（输出维度），例如 4096
          kv_dim: 输入视觉 token 的维度，例如 2100+
          heads: 多头注意力头数，要求 q_dim 能被 heads 整除
          dropout: dropout 概率
          ff_mult: FFN 内部扩展倍数
        """
        super().__init__()
        assert q_dim % heads == 0, "q_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = q_dim // heads
        self.scale = self.head_dim ** -0.5

        # 投影层，将 latent（查询）投影到 q_dim，并将输入（键和值）投影到 q_dim
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, q_dim, bias=False)

        # 两个 LayerNorm 分别用于注意力输出和 FFN 后
        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(q_dim)
        # FFN
        inner_dim = int(q_dim * ff_mult)
        self.ffn = nn.Sequential(
            nn.Linear(q_dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, q_dim, bias=False)
        )
        self.dropout = dropout

    def forward(self, latents, keys):
        # latents: (B, M, q_dim) ； keys: (B, N, kv_dim)
        B, M, _ = latents.shape
        B, N, _ = keys.shape
        # 计算查询、键和值投影
        q = self.q_proj(latents)  # (B, M, q_dim)
        k = self.k_proj(keys)  # (B, N, q_dim)
        v = self.v_proj(keys)  # (B, N, q_dim)
        # 重塑成多头形状: (B, M, q_dim) -> (B, heads, M, head_dim)
        q = q.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        q = q * self.scale

        # 跨注意力，自动启用 Flash Attention（PyTorch 2.2+）
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=False)
        attn = attn.transpose(1, 2).reshape(B, M, -1)  # (B, M, q_dim)
        attn_out = self.out_proj(attn)
        # 残差 + LayerNorm
        latents = self.norm1(latents + attn_out)
        # FFN
        ff = self.ffn(latents)
        latents = self.norm2(latents + ff)
        return latents


class PerceiverResampler(nn.Module):
    def __init__(self, q_dim, kv_dim, num_latents, depth, heads=8, dropout=0.0, ff_mult=4):
        """
        视觉压缩模块，将输入视觉 token (B, N, kv_dim) 压缩为 latent token (B, num_latents, q_dim)
        参数：
          q_dim: 输出 latent token 的维度（查询维度），例如 4096
          kv_dim: 输入视觉 token 的维度，例如 2100+
          num_latents: 压缩后 latent token 数量，例如 1
          depth: 堆叠的 ResamplerBlock 层数
          heads: 多头注意力头数
          dropout: dropout 概率
          ff_mult: FFN 内部扩展倍数
        """
        super().__init__()
        # 初始化可学习的 latent 查询，形状 (1, num_latents, q_dim)
        self.latents = nn.Parameter(torch.randn(1, num_latents, q_dim))
        # 堆叠多个 ResamplerBlock
        self.layers = nn.ModuleList([
            ResamplerBlock(q_dim, kv_dim, heads, dropout, ff_mult) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(q_dim)

    def forward(self, x):
        # x: 输入视觉 token，形状 (B, N, kv_dim)
        B = x.size(0)
        latents = self.latents.expand(B, -1, -1)  # (B, num_latents, q_dim)
        for layer in self.layers:
            latents = layer(latents, x)
        return self.norm(latents)


class FiLM(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLM, self).__init__()
        self.scale_fc = nn.Linear(condition_dim, feature_dim)
        self.shift_fc = nn.Linear(condition_dim, feature_dim)

        nn.init.zeros_(self.scale_fc.weight)
        nn.init.zeros_(self.scale_fc.bias)
        nn.init.zeros_(self.shift_fc.weight)
        nn.init.zeros_(self.shift_fc.bias)

    def forward(self, x, condition):
        # Calculate the scaling and offset parameters
        scale = self.scale_fc(condition)
        shift = self.shift_fc(condition)

        # Apply FiLM modulation
        return x * (1 + scale) + shift


class ReasoningProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReasoningProjector, self).__init__()
        self.global_query = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.attn = nn.MultiheadAttention(embed_dim=in_dim,
                                          num_heads=8,
                                          batch_first=True)
        self.mlps = nn.ModuleList([
            # nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        )

    def forward(self, x):
        B, N, C = x.shape
        q = self.global_query.expand(B, -1, -1)
        attn_out, _ = self.attn(q, x, x, need_weights=False)

        for mlp in self.mlps:
            attn_out = mlp(attn_out)
        return attn_out


class EmbodiedReasoningProjectorv2(nn.Module):
    def __init__(self, in_dim, out_dim, num_queries=3):
        super(EmbodiedReasoningProjectorv2, self).__init__()

        self.query_names = ['global', 'task',
                            'perception', 'action'][:num_queries]
        self.queries = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, in_dim)) for name in self.query_names
        })
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=8, batch_first=True)
        self.mlps = nn.ModuleList([
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ])

    def forward(self, x):
        B, N, C = x.shape
        outputs = []
        for name in self.query_names:
            q = self.queries[name].expand(B, -1, -1)  # [B, 1, C]
            attn_out, _ = self.attn(q, x, x, need_weights=False)  # [B, 1, C]
            for mlp in self.mlps:
                attn_out = mlp(attn_out)
            outputs.append(attn_out)  # [B, 1, out_dim]

        # [B, num_queries, out_dim] -> [B, 1, out_dim]
        fused = torch.mean(torch.cat(outputs, dim=1), dim=1, keepdim=True)
        return fused


class EmbodiedReasoningProjectorv3(nn.Module):
    def __init__(self, in_dim, out_dim, num_queries=3):
        super(EmbodiedReasoningProjectorv3, self).__init__()

        self.query_names = ['global', 'task',
                            'perception', 'action'][:num_queries]
        self.queries = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, in_dim)) for name in self.query_names
        })
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=8, batch_first=True)
        self.mlps = nn.ModuleList([
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ])
        self.gate = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C = x.shape
        outputs = []
        for name in self.query_names:
            q = self.queries[name].expand(B, -1, -1)  # [B, 1, C]
            attn_out, _ = self.attn(q, x, x, need_weights=False)  # [B, 1, C]
            for mlp in self.mlps:
                attn_out = mlp(attn_out)
            outputs.append(attn_out)  # [B, 1, out_dim]

        # fused = torch.mean(torch.cat(outputs, dim=1), dim=1, keepdim=True)  # [B, num_queries, out_dim] -> [B, 1, out_dim]
        stacked = torch.cat(outputs, dim=1)  # [B, num_queries, out_dim]
        gated = self.gate(stacked) * stacked  # [B, num_queries, out_dim]
        fused = gated.sum(dim=1, keepdim=True)
        return fused


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout)
        )
        self.use_residual = (in_dim == out_dim)

    def forward(self, x):
        out = self.layer(x)
        if self.use_residual:
            return out + x  # residual
        else:
            return out


class EmbodiedReasoningProjectorV4(nn.Module):
    def __init__(self, in_dim, out_dim, num_queries=2):
        super().__init__()
        self.query_names = ['task', 'perception', 'action'][:num_queries]
        self.queries = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, in_dim)) for name in self.query_names
        })
        self.norm = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=8, batch_first=True)
        self.mlp = ResidualMLP(in_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Sigmoid())

    def forward(self, x):
        B, N, C = x.shape
        outputs = []
        for name in self.query_names:
            q = self.queries[name].expand(B, -1, -1)
            attn_out, _ = self.attn(q, self.norm(
                x), self.norm(x), need_weights=False)
            attn_out = self.mlp(attn_out)
            outputs.append(attn_out)

        stacked = torch.cat(outputs, dim=1)
        gated = self.gate(stacked) * stacked + stacked  # gated residual
        fused = gated.sum(dim=1, keepdim=True)
        return fused


class CognitionReasoningProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CognitionReasoningProjector, self).__init__()
        # self.global_query = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=8, batch_first=True)
        self.mlps = nn.ModuleList([
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        )

    def forward(self, cognition, x):
        B, N, C = x.shape
        q = cognition
        attn_out, _ = self.attn(q, x, x, need_weights=False)

        for mlp in self.mlps:
            attn_out = mlp(attn_out)
        return attn_out


class CoTReasoningBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, cos_sin=None):
        x = hidden_states + input_injection
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class HierarchicalCoTUpdater(nn.Module):
    def __init__(self, dim, task_layers=1, action_layers=1, task_cycles=2, action_cycles=2):
        super().__init__()
        self.task_cycles = task_cycles
        self.action_cycles = action_cycles

        self.task_layer = nn.ModuleList([CoTReasoningBlock(dim) for _ in range(task_layers)])
        self.action_layer = nn.ModuleList([CoTReasoningBlock(dim) for _ in range(action_layers)])

    def forward(self, zH, zL, reaoning_features):
        with torch.no_grad():
            for _H_step in range(self.task_cycles):
                for _L_step in range(self.action_cycles):
                    if not ((_H_step == self.task_cycles - 1) and (_L_step == self.action_cycles - 1)):
                        for block in self.action_layer:
                            zL = block(zL, zH + reaoning_features)

                if not (_H_step == self.task_cycles - 1):
                    for block in self.task_layer:
                        zH = block(zH, zL)

        # 最后一步参与梯度传播
        for block in self.action_layer:
            zL = block(zL, zH + reaoning_features)
        for block in self.task_layer:
            zH = block(zH, zL)

        return zH, zL


class HiCoTWrapper(nn.Module):
    def __init__(self, dim, task_cycles=1, action_cycles=2):
        super().__init__()
        self.hicot = HierarchicalCoTUpdater(dim, task_cycles=task_cycles, action_cycles=action_cycles)
        self.task_init = nn.Parameter(torch.zeros(1, 1, dim))
        self.action_init = nn.Parameter(torch.zeros(1, 1, dim))
        self.global_query = nn.Parameter(torch.zeros(1, 1, dim))

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.mlps = nn.ModuleList([
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        ]
        )

    def forward(self, x):
        B, N, C = x.shape
        q = self.global_query.expand(B, -1, -1)
        attn_out, _ = self.attn(q, x, x, need_weights=False)
        for mlp in self.mlps:
            attn_out = mlp(attn_out)

        zH = self.task_init.expand(B, -1, -1).clone()
        zL = self.action_init.expand(B, -1, -1).clone()
        zH, zL = self.hicot(zH, zL, attn_out)

        return zH


class HierarchicalReasoningProjector(nn.Module):
    """
    分层推理 Projector，zH/zL可跨推理步保留，增强时序记忆。
    """
    def __init__(self, dim, H_layers=1, L_layers=1, H_cycles=2, L_cycles=2):
        super().__init__()
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.H_level = nn.ModuleList([CoTReasoningBlock(dim) for _ in range(H_layers)])
        self.L_level = nn.ModuleList([CoTReasoningBlock(dim) for _ in range(L_layers)])
        self.norm = nn.LayerNorm(dim)
        self.zH = None
        self.zL = None

        self.global_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.mlps = nn.ModuleList([
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        ]
        )

    def reset_state(self):
        self.zH = None
        self.zL = None

    def forward(self, x):
        # x: [B, T, D]，B为时序长度（如32），T通常为1或token数
        B, T, D = x.shape
        device = x.device

        q = self.global_query.expand(B, -1, -1)
        attn_out, _ = self.attn(q, x, x, need_weights=False)
        for mlp in self.mlps:
            attn_out = mlp(attn_out)
        if self.zH is None and self.zL is None:
            zH = torch.zeros(1, 1, D, device=device)
            zL = torch.zeros(1, 1, D, device=device)
        else:
            zH = self.zH
            zL = self.zL
        for t in range(B):
            xt = attn_out[t:t+1]  # [1, T, D]
            with torch.no_grad():
                for _ in range(self.H_cycles):
                    for _ in range(self.L_cycles):
                        if not ((_ == self.H_cycles - 1) and (_ == self.L_cycles - 1)):
                            for l_layer in self.L_level:
                                zL = l_layer(zL, zH + xt)
                    if not (_ == self.H_cycles - 1):
                        for h_layer in self.H_level:
                            zH = h_layer(zH, zL)
            for l_layer in self.L_level:
                zL = l_layer(zL, zH + xt)
            for h_layer in self.H_level:
                zH = h_layer(zH, zL)
        self.zH = zH.detach()
        self.zL = zL.detach()

        pooled = self.norm(zH)
        return pooled


class CoTMemoryBank:
    def __init__(self, expire_threshold: int = 6, using_attention: bool = False, num_layers: int = 2, feature_dim: int = 4096):
        self.cot_dict = {}
        self.tags = get_cot_tags_list()
        self.update_counter = {tag: 0 for tag in self.tags}  # 记录每个tag未更新次数
        self.expire_threshold = expire_threshold  # 超过5次未更新则丢弃
        self.using_attention = using_attention  # 是否使用注意力机制
        if using_attention:
            print('using attention for CoT memory bank')
            self.cross_blocks = nn.ModuleList([
                CrossTransformerBlock(feature_dim)
                for _ in range(num_layers)
            ])
        self.reset()

    def update_cot_embedding(self, decoded: str, reasoning_feats: torch.Tensor):
        """
        decoded: str, 解析出的CoT文本（如 "...TASK: ... PLAN: ... MOVE: ..."）
        reasoning_feats: torch.Tensor, shape [B, T, D]，每个字段对应一个T
        """
        # 1. 找到每个tag在字符串中的起始位置
        tags_sorted = sorted(self.tags, key=lambda x: -len(x))
        tag_pos = []
        used = [0] * len(decoded)  # 标记已被tag占用的字符
        for tag in tags_sorted:
            for m in re.finditer(re.escape(tag), decoded):
                idx = m.start()
                # 检查该区域是否已被占用
                if any(used[idx:idx+len(tag)]):
                    continue
                tag_pos.append((idx, tag))
                for i in range(idx, idx+len(tag)):
                    used[i] = 1
        # 2. 按出现顺序排序
        tag_pos.sort()
        # 3. 遍历每个tag，更新embedding
        updated_tags = set()
        for i, (idx, tag) in enumerate(tag_pos):
            # 计算下一个tag的起始位置，或到字符串结尾
            if i + 1 < len(tag_pos):
                next_idx = tag_pos[i + 1][0]
            else:
                next_idx = reasoning_feats.shape[1]
                # reasoning_feats 的顺序应与tag出现顺序一致
            content_len = next_idx - (idx)
            self.cot_dict[tag] = reasoning_feats[:, idx:idx+content_len, :]  # [B, 1, D]
            self.cot_dict[tag] = reasoning_feats[:, idx:idx+content_len, :]  # [B, 1, D]
            self.update_counter[tag] = 0  # 本次更新，计数归零
            updated_tags.add(tag)

        # 未更新的tag计数+1，过期则丢弃（TASK和PLAN除外）
        for tag in self.tags:
            if tag not in updated_tags and tag in self.cot_dict and tag not in []:  # [CotTag.TASK.value, CotTag.PLAN.value]:
                self.update_counter[tag] += 1
                if self.update_counter[tag] > self.expire_threshold:
                    del self.cot_dict[tag]
                    self.update_counter[tag] = 0

    def update_cot_embedding2(self, decoded: str, reasoning_feats: torch.Tensor, tokenizer=None):

        assert tokenizer is not None, "You must provide a tokenizer for correct alignment!"

        token_ids = tokenizer.encode(decoded, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        char_offsets = []
        idx = 0
        for tok in tokens:
            while idx < len(decoded) and decoded[idx].isspace():
                idx += 1
            pos = decoded.find(tok.replace('▁', ' ').strip(), idx)
            if pos == -1:
                pos = idx
            char_offsets.append(pos)
            idx = pos + len(tok.replace('▁', ' ').strip())

        tags_sorted = sorted(self.tags, key=lambda x: -len(x))
        tag_token_pos = []
        used = [0] * len(tokens)
        for tag in tags_sorted:
            tag_tokens = tokenizer.encode(tag, add_special_tokens=False)
            tag_len = len(tag_tokens)
            for i in range(len(tokens) - tag_len + 1):
                if used[i:i+tag_len].count(1) > 0:
                    continue
                if tokens[i:i+tag_len] == tokenizer.convert_ids_to_tokens(tag_tokens):
                    tag_token_pos.append((i, tag))
                    for j in range(i, i+tag_len):
                        used[j] = 1
        tag_token_pos.sort()

        updated_tags = set()
        for i, (idx, tag) in enumerate(tag_token_pos):
            if i + 1 < len(tag_token_pos):
                next_idx = tag_token_pos[i + 1][0]
            else:
                next_idx = reasoning_feats.shape[1]
            content_len = next_idx - idx
            self.cot_dict[tag] = reasoning_feats[:, idx:idx+content_len, :]  # [B, N, D]
            self.update_counter[tag] = 0
            updated_tags.add(tag)

        for tag in self.tags:
            if tag not in updated_tags and tag in self.cot_dict and tag not in [CotTag.TASK.value, CotTag.PLAN.value]:
                self.update_counter[tag] += 1
                if self.update_counter[tag] > self.expire_threshold:
                    del self.cot_dict[tag]
                    self.update_counter[tag] = 0

    def get_cot_embedding(self) -> torch.tensor:
        """
        返回所有已存储的CoT embedding，拼接为 [B, N, D]，N为已存字段数
        """
        if not self.cot_dict:
            return None
        # 按tag顺序拼接
        emb_list = [self.cot_dict[tag] for tag in self.tags if tag in self.cot_dict]
        emb_list = [self.cot_dict[tag] for tag in self.tags if tag in self.cot_dict]
        if emb_list:
            return torch.cat(emb_list, dim=1)  # [B, N, D]
        else:
            return None

    def get_cot_embedding_with_query(self, cognition_features: torch.tensor) -> torch.tensor:
        B, token_num, D = cognition_features.shape
        outputs = []
        for i in range(B):
            query = cognition_features[i].unsqueeze(0)  # (1, token_num, D)
            cot_hist = self.get_cot_embedding()
            if cot_hist:
                cot_memory = torch.cat(
                    cot_hist, dim=0).unsqueeze(0)  # (1, M, D)
                x = query
                for block in self.cross_blocks:
                    # cross + ffn
                    x = block(x, cot_memory)
                attn_out = x
            else:
                attn_out = query
            fused = query + attn_out
            outputs.append(fused.squeeze(0).unsqueeze(0))
        return torch.cat(outputs, dim=0)  # (B, token_num, D)

    def reset(self):
        self.cot_dict = {}
        self.update_counter = {tag: 0 for tag in self.tags}


class CogACT(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_model_type: str = 'DiT-B',
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        timestep_sample="uniform",
        traj_group_size=1,
        use_memory_bank=False,
        memory_bank_version='v1',
        use_img_res=False,
        img_res_share_vision_encoder=True,
        image_resize_strategy="resize-naive",
        load_proprio=False,
        proprio_to_vlm=False,
        lang_inject="no",
        lang_action_out: Optional[bool] = False,
        use_cot: Optional[bool] = False,
        use_cot_trigger: Optional[bool] = False,
        use_moe: Optional[bool] = False,
        use_cot_memory: Optional[bool] = False,
        use_cot_memory_attn: Optional[bool] = False,
        cot_frozen_step: Optional[int] = 0,
        cot_memory_expire: Optional[int] = 6,
        **kwargs,
    ) -> None:
        super().__init__()

        self.use_img_res = use_img_res
        num_extra_token = 0
        if self.use_img_res:
            num_extra_token += 1

        self.load_proprio = load_proprio
        self.proprio_to_vlm = proprio_to_vlm
        if self.load_proprio or self.proprio_to_vlm:
            print('WARNING: Only support bridge proprio!!!!!!')
            self.proprio_encoder = nn.Sequential(
                nn.Linear(8, token_size),
                nn.SiLU(),
                nn.Linear(token_size, token_size),
            )
            if self.load_proprio:
                num_extra_token += 1
            if self.proprio_to_vlm:
                self.proprio_projector = nn.Sequential(
                    nn.LayerNorm(token_size),
                    nn.Linear(token_size, token_size),
                    nn.GELU(),
                    nn.Linear(token_size, token_size),
                )

        self.action_model = ActionModel(model_type=action_model_type,
                                        token_size=token_size,
                                        in_channels=action_dim,
                                        future_action_window_size=future_action_window_size,
                                        past_action_window_size=past_action_window_size,
                                        timestep_sample=timestep_sample,
                                        num_extra_token=num_extra_token,
                                        )
        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ['action_model', 'ema_diffusion']
        else:
            self.all_module_keys = ['action_model']
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)

        # Diffusion head is always trainable
        self._trainable_module_keys = ['action_model']
        self.norm_stats = norm_stats

        self.traj_group_size = traj_group_size
        self.use_memory_bank = use_memory_bank

        if use_memory_bank:
            assert self.traj_group_size > 1, "Memory bank only supports traj_group_size > 1"
            if memory_bank_version == 'v1':
                self.memory_bank = MemoryBank(feature_dim=token_size, traj_group_size=traj_group_size, max_memory_size=128)
            elif memory_bank_version == 'v2':
                self.memory_bank = MemoryBankV2(feature_dim=token_size, traj_group_size=traj_group_size, max_memory_size=128)
            else:
                raise NotImplementedError(
                    f"Memory bank version {memory_bank_version} not implemented")

        self.img_res_share_vision_encoder = img_res_share_vision_encoder
        if self.use_img_res:
            kv_dim = self.vlm.vision_backbone.dino_featurizer.patch_embed.proj.weight.shape[
                0] + self.vlm.vision_backbone.siglip_featurizer.patch_embed.proj.weight.shape[0]
            if not self.img_res_share_vision_encoder:
                self.img_res_encoder, _ = get_vision_backbone_and_transform(
                    vision_backbone_id='dinov2-vit-l',
                    image_resize_strategy=image_resize_strategy,
                )
                kv_dim = self.vlm.vision_backbone.dino_featurizer.patch_embed.proj.weight.shape[
                    0]

            self.perceiver_resampler = PerceiverResampler(q_dim=token_size, kv_dim=kv_dim, num_latents=1, depth=2, heads=8, dropout=0.0, ff_mult=4)

        self.lang_action_out = lang_action_out
        self.use_cot = use_cot
        self.lang_inject = lang_inject
        if self.lang_inject:
            print(
                f'-----------using lang_inject {self.lang_inject} for reasoning of VLA model-------------')
            assert lang_action_out or use_cot
            if self.lang_inject == 'v2':
                self.reasoning_projector = EmbodiedReasoningProjectorv2(
                    token_size, token_size, num_queries=4)
            elif self.lang_inject == 'v3':
                self.reasoning_projector = EmbodiedReasoningProjectorv3(
                    token_size, token_size, num_queries=4)
            elif self.lang_inject == 'v1':
                self.reasoning_projector = ReasoningProjector(
                    token_size, token_size)
            elif self.lang_inject == 'v4':
                self.reasoning_projector = EmbodiedReasoningProjectorV4(
                    token_size, token_size, num_queries=3)
            elif self.lang_inject == 'cognition':
                self.reasoning_projector = CognitionReasoningProjector(
                    token_size, token_size)
            elif self.lang_inject == 'hicot':
                self.reasoning_projector = HiCoTWrapper(token_size)
            elif self.lang_inject == 'hicot_v2':
                self.reasoning_projector = HierarchicalReasoningProjector(token_size)
            else:
                self.reasoning_projector = None
            print(
                f'-----------using reasoning_projector {self.reasoning_projector} for reasoning of VLA model-------------')
            self.reasoning_film = FiLM(token_size, token_size)

        self.use_moe = use_moe
        if self.use_moe:
            print('-----------using moe for reasoning of VLA model-------------')
            from st_moe_pytorch import MoE, SparseMoEBlock
            moe = MoE(
                dim=token_size,
                num_experts=4,
                gating_top_n=2,
                threshold_train=0.2,
                threshold_eval=0.2,
                capacity_factor_train=1.25,
                capacity_factor_eval=2.,
                balance_loss_coef=1e-2,
                router_z_loss_coef=1e-3,
                allow_var_seq_len=True
            )
            self.moe_block = SparseMoEBlock(
                moe,
                add_ff_before=True,
                add_ff_after=True
            )
            self.reasoning_projector = ReasoningProjector(
                token_size, token_size)
            self.reasoning_film = FiLM(token_size, token_size)

        self.base_prompt = ''  # f"{CotTag.TASK.value}"  # 初始化时可根据实际情况赋值
        self.frozen_prompt = None
        self.max_freezing_time = cot_frozen_step  # 可根据需要调整
        print('------------max_freezing_time: ', self.max_freezing_time, '-----------------')
        self.time_frozen = 0

        self.use_cot_memory = use_cot_memory
        self.cot_memory_expire = cot_memory_expire
        if self.use_cot_memory:
            print(
                f'-----------self.use_cot_memory: {self.use_cot_memory}, self.cot_memory_expire: {self.cot_memory_expire}-------------')
            self.use_cot_memory_attn = use_cot_memory_attn
            self.cot_memory_bank = CoTMemoryBank(
                expire_threshold=self.cot_memory_expire, using_attention=use_cot_memory_attn, num_layers=2, feature_dim=token_size)

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys

    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone

    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone

    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks=None,
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        if self.load_proprio or self.proprio_to_vlm:
            proprio_emb = self.proprio_encoder(proprio)

        if self.proprio_to_vlm:
            proprio_emb_vlm = self.proprio_projector(proprio_emb)

        output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            proprio_feat=proprio_emb_vlm if self.proprio_to_vlm else None,
        )

        # extract the last hidden state and the learnable EOS token feature
        last_hidden = output.hidden_states[-1]

        # extract the visual token number
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        if self.proprio_to_vlm:
            num_patch += 1

        last_hidden = last_hidden[:, num_patch:]

        # extract the cognition feature
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]

        # reasoning inject
        if self.lang_inject:
            if self.lang_inject == 'hicot_v2':
                self.reasoning_projector.reset_state()
            max_len = int(attention_mask.sum(dim=1).max().item())
            masked_hidden = last_hidden * attention_mask.unsqueeze(-1)
            reasoning_feats = masked_hidden[:, :max_len, :]
            reasoning_feats = reasoning_feats[:, 1:-1, :]
            reasoning_feats = self.reasoning_projector(reasoning_feats)
            reasoning_feats = self.reasoning_projector(reasoning_feats)
            cognition_features = self.reasoning_film(cognition_features, reasoning_feats)

        if self.use_moe:
            max_len = int(attention_mask.sum(dim=1).max().item())
            masked_hidden = last_hidden * attention_mask.unsqueeze(-1)
            reasoning_feats = masked_hidden[:, :max_len, :]
            reasoning_feats = reasoning_feats[:, 1:-1, :]  # [B,T,D]
            moe_reasoning_feats, total_aux_loss, _, _ = self.moe_block(reasoning_feats)  # [B,T,D]
            reasoning_feats = self.reasoning_projector(moe_reasoning_feats)  # [B,1,D]
            cognition_features = self.reasoning_film(cognition_features, reasoning_feats)  # [B,1,D]

        actions_future = actions[:, -(self.future_action_window_size+1):, :]

        if self.use_img_res:
            if self.img_res_share_vision_encoder:
                raw_img_feat = self.vlm.current_raw_img_feat
            else:
                raw_img_feat = self.img_res_encoder(pixel_values['dino'])
            img_res_feat = self.perceiver_resampler(raw_img_feat)

            cognition_features = torch.cat([cognition_features, img_res_feat], dim=1)

        if self.load_proprio:
            cognition_features = torch.cat([cognition_features, proprio_emb], dim=1)

        if self.use_memory_bank:
            self.memory_bank.reset()
            cognition_features = self.memory_bank.process_batch(cognition_features)

        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(repeated_diffusion_steps, 1, 1)  # [repeated_diffusion_steps*B, 1, D]

        loss = self.action_model.loss(actions_repeated, cognition_features_repeated)

        if self.use_moe:
            loss += total_aux_loss

        return loss, output

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector,
                            MLPProjector, FusedMLPProjector, DiT},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: str = 'DiT-B',
        use_ema: bool = False,
        norm_stats=None,
        **kwargs,
    ) -> CogACT:

        # Load VLM backbone, borrowed from PrismaticVLM
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(
            pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(
                model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize CogACT
        cogact = CogACT(vlm,
                        token_size=vlm.llm_backbone.llm.lm_head.in_features,
                        action_dim=action_dim,
                        future_action_window_size=future_action_window_size,
                        past_action_window_size=past_action_window_size,
                        action_model_type=action_model_type,
                        use_ema=use_ema,
                        norm_stats=norm_stats,
                        **kwargs,
                        )

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            num_extra_token = 0
            if 'use_img_res' in kwargs and kwargs['use_img_res']:
                num_extra_token += 1
            if 'load_proprio' in kwargs and kwargs['load_proprio']:
                num_extra_token += 1

            if num_extra_token > 0:
                pretrained_pe = model_state_dict["action_model"]["net.positional_embedding"]
                if cogact.action_model.net.positional_embedding.data.shape[0] != pretrained_pe.shape[0]:
                    assert cogact.action_model.net.positional_embedding.data.shape[
                        0] > pretrained_pe.shape[0]
                    cur_pe = cogact.action_model.net.positional_embedding.data
                    cur_pe[num_extra_token:] = pretrained_pe
                    model_state_dict["action_model"]["net.positional_embedding"] = cur_pe

                # 用于目前带bug训练出的模型
                model_uncondition_num = cogact.action_model.net.z_embedder.uncondition.shape[
                    0]
                ckpt_uncondition_num = model_state_dict["action_model"]['net.z_embedder.uncondition'].shape[0]
                if model_uncondition_num != ckpt_uncondition_num:
                    print(
                        'WARNING: model_uncondition_num != ckpt_uncondition_num, we use boardcasting!!!!!!')
                    model_state_dict["action_model"]['net.z_embedder.uncondition'] = model_state_dict[
                        "action_model"]['net.z_embedder.uncondition'].expand(model_uncondition_num, -1)

            cogact.action_model.load_state_dict(
                model_state_dict["action_model"])
            assert use_ema is False, "Does not support using EMA weights from pretrained checkpoint."
            if "ema_diffusion" in model_state_dict and use_ema:
                cogact.ema_diffusion.load_state_dict(
                    model_state_dict["ema_diffusion"])
            elif use_ema:
                cogact.ema_diffusion.load_state_dict(
                    model_state_dict["action_model"])
        else:
            overwatch.warning(
                "No ActionModel found in the pretrained checkpoint. Initializing a new one.")
        return cogact

    @torch.inference_mode()
    def predict_action(
        self, image: Image,
        instruction: str,
        proprio: Optional[torch.Tensor] = None,
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        reset_memory: bool = False,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
        was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(
            role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        lang_action_len = self.get_action_dim(unnorm_key) * (self.future_action_window_size + 1)
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
            if self.use_cot is False and self.lang_action_out is False:
                max_new_tokens = 1
            elif self.use_cot:
                max_new_tokens = 1024
            elif self.lang_action_out:
                max_new_tokens = lang_action_len + 2
        else:
            raise ValueError(
                f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(
                self.vlm.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(
                f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            # fmt: on

        # language out
        if self.use_cot or self.lang_action_out:
            from prismatic.vla.action_tokenizer import ActionTokenizer
            seq_ids = output.sequences[0].cpu().tolist()
            hf_tok = self.llm_backbone.tokenizer
            action_tok = ActionTokenizer(hf_tok)
            b = action_tok.action_token_begin_idx
            e = b + action_tok.n_bins
            decoded, buf = [], []
            for tid in seq_ids:
                if b < tid <= e:
                    if buf:
                        decoded.append(hf_tok.decode(
                            buf, skip_special_tokens=False))
                        buf.clear()
                    decoded.append(action_tok.decode_token_ids_to_actions(
                        np.array([tid], int))[0])
                else:
                    buf.append(tid)
            if buf:
                decoded.append(hf_tok.decode(buf, skip_special_tokens=False))
            print("Langage:", decoded)

        model_dtype = next(self.action_model.net.parameters()).dtype
        # cognition_features = output.hidden_states[-1][-1][:,-1,:]
        cognition_features = output.hidden_states[0][-1][:, -1, :]
        assert (cognition_features.shape[0], cognition_features.shape[1]) == (
            1, 4096), "Batch size must be 1 for action prediction"
        cognition_features = cognition_features.unsqueeze(
            1).to(model_dtype)  # [B, 1, D]

        # reasoning inject
        if self.lang_inject:
            reasoning_feats = []
            for i in range(1, len(output.hidden_states)):
                reasoning_feats.append(output.hidden_states[i][-1])

            if reasoning_feats != []:
                reasoning_feats = torch.cat(
                    reasoning_feats, dim=1)  # [B, T, D]
                reasoning_feats = self.reasoning_projector(
                    reasoning_feats, visualize_attention=False)
                cognition_features = self.reasoning_film(
                    cognition_features, reasoning_feats)

        if self.use_img_res:
            if self.img_res_share_vision_encoder:
                raw_img_feat = self.vlm.current_raw_img_feat
            else:
                raw_img_feat = self.img_res_encoder(pixel_values['dino'])

            img_res_feat = self.perceiver_resampler(raw_img_feat)
            cognition_features = torch.cat(
                [cognition_features, img_res_feat], dim=1)

        if self.load_proprio:
            proprio_emb = self.proprio_encoder(proprio)
            cognition_features = torch.cat(
                [cognition_features, proprio_emb], dim=1)

        if self.use_memory_bank:
            if reset_memory:
                print(" ** reset memory ** ")
                self.memory_bank.reset()
            cognition_features = self.memory_bank.process_batch(
                cognition_features)

        # Sample random noise
        B = cognition_features.shape[0]
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels,
                            device=cognition_features.device).to(model_dtype)  # [B, T, D]

        # Setup classifier-free guidance:
        using_cfg = cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  # [k, D]
            uncondition = uncondition.expand(
                B, *uncondition.shape[1:])  # [B, k, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn,
                                                                        noise.shape,
                                                                        noise,
                                                                        clip_denoised=False,
                                                                        model_kwargs=model_kwargs,
                                                                        progress=False,
                                                                        device=cognition_features.device,
                                                                        eta=0.0
                                                                        )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn,
                                                                noise.shape,
                                                                noise,
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device
                                                                )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(
            action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(
            action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(
            normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) *
            (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions

    @torch.inference_mode()
    def predict_action_cot(
        self, image: Image,
        instruction: str,
        proprio: Optional[torch.Tensor] = None,
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        reset_memory: bool = False,
        cot_version: str = '',
        timestep: int = 0,
        allow_prefix: bool = False,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

        chosen_cot_tag = ''
        # chosen_cot_tag = CotTag.SUBTASK.value
        prompt_builder = self.vlm.get_prompt_builder()
        if cot_version == 'v4':
            if chosen_cot_tag == '':
                prompt_builder.add_turn(
                    role="human", message=f"What action should the robot take to {instruction.lower()}?")
            else:
                prompt_builder.add_turn(
                    role="human", message=f"What action should the robot take to {instruction.lower()} based on {chosen_cot_tag.lower()[:-1]}?")
        else:
            prompt_builder.add_turn(
                role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        if self.time_frozen <= 0 or reset_memory:
            self.frozen_prompt = self.base_prompt
            self.time_frozen = self.max_freezing_time
            cot_prompt = chosen_cot_tag
        else:
            cot_prompt = ''
        self.time_frozen -= 1

        lang_action_len = self.get_action_dim(unnorm_key) * (self.future_action_window_size + 1)
        lang_action_len = self.get_action_dim(unnorm_key) * (self.future_action_window_size + 1)
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if cot_version == 'v1' or cot_version == 'v2':
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
            input_ids = torch.cat((input_ids, tokenizer(
                self.frozen_prompt, return_tensors="pt").input_ids.to(self.vlm.device)[:, 1:],), dim=1)

        elif cot_version == 'v3' or cot_version == 'v3.2' or cot_version == 'v4' or cot_version == 'v5' or cot_version == 'v6':
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
            input_ids = torch.cat((input_ids, tokenizer(cot_prompt, return_tensors="pt").input_ids.to(self.vlm.device)[:, 1:],), dim=1)
            input_ids = torch.cat((input_ids, tokenizer(self.frozen_prompt, return_tensors="pt").input_ids.to(self.vlm.device)[:, 1:],), dim=1)
            input_ids = torch.cat((input_ids, tokenizer(cot_prompt, return_tensors="pt").input_ids.to(self.vlm.device)[:, 1:],), dim=1)
            input_ids = torch.cat((input_ids, tokenizer(self.frozen_prompt, return_tensors="pt").input_ids.to(self.vlm.device)[:, 1:],), dim=1)

        if self.use_cot is False and self.lang_action_out is False:
            max_new_tokens = 1
        elif self.use_cot:
            max_new_tokens = 1024
        elif self.lang_action_out:
            max_new_tokens = lang_action_len + 2

        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(
                self.vlm.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(
                f"Unsupported `pixel_values` type = {type(pixel_values)}")

        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            def only_eos_prefix(batch_id, input_ids):
                return [tokenizer.eos_token_id]
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=only_eos_prefix if (allow_prefix or len(self.frozen_prompt) != 0) else None,
                **kwargs,
            )
        if self.use_cot or self.lang_action_out:
            from prismatic.vla.action_tokenizer import ActionTokenizer
            seq_ids = output.sequences[0].cpu().tolist()
            hf_tok = self.llm_backbone.tokenizer
            action_tok = ActionTokenizer(hf_tok)
            b = action_tok.action_token_begin_idx
            e = b + action_tok.n_bins
            decoded, buf = [], []
            for tid in seq_ids:
                if b < tid <= e:
                    if buf:
                        decoded.append(hf_tok.decode(
                            buf, skip_special_tokens=False))
                        buf.clear()
                    decoded.append(action_tok.decode_token_ids_to_actions(
                        np.array([tid], int))[0])
                else:
                    buf.append(tid)
            if buf:
                decoded.append(hf_tok.decode(buf, skip_special_tokens=False))
            output_decoded = tokenizer.decode(seq_ids[-len(output.hidden_states):])
            print(f"Timestep: {timestep}, New Generated CoT length: {len(output.hidden_states)}, {output_decoded}")

        debug = False
        if debug:
            score_tensor = output.scores[0]  # shape: (batch_size, vocab_size)
            prob_tensor = F.softmax(score_tensor, dim=-1)
            topk = torch.topk(prob_tensor[0], k=10)  # top 10
            top_probs = topk.values   # tensor of shape (10,)
            top_indices = topk.indices  # tensor of shape (10,)
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
            for i in range(10):
                token_str = top_tokens[i]
                prob = top_probs[i].item()
                print(f"{i+1:2d}. Token: {token_str:15}  |  Prob: {prob:.6f}")

        model_dtype = next(self.action_model.net.parameters()).dtype
        cognition_features = output.hidden_states[0][-1][:, -1, :]
        # cognition_features = output.hidden_states[-1][-1][:, -1, :]
        assert (cognition_features.shape[0], cognition_features.shape[1]) == (1, 4096), "Batch size must be 1 for action prediction"
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        if self.use_cot_memory:
            if reset_memory:
                print(" ** reset cot memory bank ** ")
                self.cot_memory_bank.reset()
                if self.lang_inject == 'hicot_v2':
                    self.reasoning_projector.reset_state()
                    print("** reset inject-hicotv2 hidden states **")
            reasoning_feats = []
            for i in range(1, len(output.hidden_states)):
                reasoning_feats.append(output.hidden_states[i][-1])
            if reasoning_feats != []:
                reasoning_feats = torch.cat(reasoning_feats, dim=1)  # [B, T, D]
                output_decoded = tokenizer.decode(seq_ids[-len(output.hidden_states):])
                # self.cot_memory_bank.update_cot_embedding(output_decoded, reasoning_feats)
                self.cot_memory_bank.update_cot_embedding2(output_decoded, reasoning_feats, tokenizer)

        if self.lang_inject != 'no':
            if self.use_cot_memory:
                reasoning_feats = self.cot_memory_bank.get_cot_embedding()
                if reasoning_feats is not None:
                    reasoning_feats = self.reasoning_projector(reasoning_feats)
                    reasoning_feats = self.reasoning_projector(reasoning_feats)
                    cognition_features = self.reasoning_film(cognition_features, reasoning_feats)
            else:
                reasoning_feats = []
                for i in range(1, len(output.hidden_states)):
                    reasoning_feats.append(output.hidden_states[i][-1])
                if reasoning_feats != []:
                    reasoning_feats = torch.cat(reasoning_feats, dim=1)  # [B, T, D]
                    reasoning_feats = self.reasoning_projector(reasoning_feats)
                    reasoning_feats = self.reasoning_projector(reasoning_feats)
                    cognition_features = self.reasoning_film(cognition_features, reasoning_feats)

        if self.use_img_res:
            if self.img_res_share_vision_encoder:
                raw_img_feat = self.vlm.current_raw_img_feat
            else:
                raw_img_feat = self.img_res_encoder(pixel_values['dino'])
            img_res_feat = self.perceiver_resampler(raw_img_feat)
            cognition_features = torch.cat(
                [cognition_features, img_res_feat], dim=1)

        if self.load_proprio:
            proprio_emb = self.proprio_encoder(proprio)
            cognition_features = torch.cat(
                [cognition_features, proprio_emb], dim=1)

        if self.use_memory_bank:
            if reset_memory:
                print(" ** reset memory ** ")
                self.memory_bank.reset()
            cognition_features = self.memory_bank.process_batch(
                cognition_features)

        # Sample random noise
        B = cognition_features.shape[0]
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  # [B, T, D]
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  # [B, T, D]

        # Setup classifier-free guidance:
        using_cfg = cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  # [k, D]
            uncondition = uncondition.expand(
                B, *uncondition.shape[1:])  # [B, k, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn,
                                                                        noise.shape,
                                                                        noise,
                                                                        clip_denoised=False,
                                                                        model_kwargs=model_kwargs,
                                                                        progress=False,
                                                                        device=cognition_features.device,
                                                                        eta=0.0
                                                                        )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn,
                                                                noise.shape,
                                                                noise,
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device
                                                                )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) *
            (action_high - action_low) + action_low,
            normalized_actions,
        )

        decoded_tokens = tokenizer.decode(output.sequences[0], skip_special_tokens=False)
        decoded_tokens = tokenizer.decode(output.sequences[0], skip_special_tokens=False)
        if "\nOut: " in decoded_tokens:
            prompt_out = decoded_tokens.split("\nOut: ")[-1]
        else:
            prompt_out = decoded_tokens

        if cot_version == 'v2' or cot_version == 'v1':
            prompt_out = decoded_tokens.split("</s>")[1]
            prompt_out += "</s>"
        if cot_version == 'v3' or cot_version == 'v3.2' or cot_version == 'v4' or cot_version == 'v5' or cot_version == 'v6':
            prompt_out = prompt_out.split("</s>")[1]
            prompt_out += "</s>"
        self.frozen_prompt = prompt_out.strip()

        return actions, normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
