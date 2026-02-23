"""This file contains implementation for MaskGIT model.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/huggingface/open-muse
    https://github.com/baaivision/MUSE-Pytorch
    https://github.com/sail-sg/MDT/blob/main/masked_diffusion/models.py
"""

import torch
from torch import nn
import numpy as np
import math
import torch.utils.checkpoint
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder
from einops import rearrange
import torch.nn.functional as F
from functools import partial

import json
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from pathlib import Path

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import UViTBlock

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class CausalAttention(nn.Module):
    """
    官方代码移植: 带 QK Norm 的因果注意力
    Reference: tokenbridge.py CausalAttention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 🌟 关键点 1: QK Norm (防止 Attention Score 极大导致梯度爆炸)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # 应用 QK Norm
        q, k = self.q_norm(q), self.k_norm(k)

        # 使用 PyTorch 原生 SDPA 加速
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CausalBlock(nn.Module):
    """
    官方代码移植: AdaLN Transformer Block
    Reference: tokenbridge.py CausalBlock
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_norm=True, 
                 proj_drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm2 = norm_layer(dim)
        
        # MLP 部分
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_drop)
        )

        # 🌟 关键点 2: AdaLN Modulation (根据 Backbone 特征 z 生成 shift/scale/gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        # c 是 Condition (来自 Backbone)
        # 将 c (B*L, D) 映射为调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Modulation 逻辑: x = x + gate * Block(Norm(x) * scale + shift)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    官方代码移植: 输出前的最终层
    Reference: tokenbridge.py FinalLayer
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # 🌟 关键点 3: Frozen Norm (elementwise_affine=False)
        self.norm_final = norm_layer(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim)
        )
    
    def forward(self, x, c):
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x

# =========================================================================
# 👇 2. 全新重写的 LocalARHead (使用上述 AdaLN 模块)
# =========================================================================

class LocalARHead(nn.Module):
    """
    TokenBridge 风格的 Local AR Head (完全替换旧版)
    """
    def __init__(self, hidden_size, codebook_size, augment_ratio, num_layers=2, nhead=8):
        super().__init__()
        self.augment_ratio = augment_ratio
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size

        # 1. 基础嵌入
        self.input_emb = nn.Embedding(codebook_size + 1, hidden_size)
        self.sos_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        # 2. 位置编码 (用于区分 AR 步骤 0~3)
        self.pos_emb = nn.Parameter(torch.randn(1, augment_ratio, hidden_size) * 0.02)
        self.cond_pos_emb = nn.Parameter(torch.randn(1, augment_ratio, hidden_size) * 0.02)
        # 3. 条件投影: 将 Backbone 特征 z 映射到 AdaLN 所需维度
        # 对应 tokenbridge.py 中的 self.condition_proj
        self.condition_proj = nn.Linear(hidden_size, hidden_size)

        # 4. 核心 Blocks (使用官方 CausalBlock)
        self.blocks = nn.ModuleList([
            CausalBlock(
                dim=hidden_size, 
                num_heads=nhead, 
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_norm=True, # 开启 QK Norm
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for _ in range(num_layers)
        ])
        
        # 5. 最终层 (使用官方 FinalLayer)
        self.final_layer = FinalLayer(hidden_size, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        # 6. 输出头 (使用 ModuleList 实现独立 Linear，解决优化冲突)
        # 对应 tokenbridge.py 中的 self.channel_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, codebook_size)
            for _ in range(augment_ratio)
        ])
        
        # 7. 初始化 (至关重要)
        self.initialize_weights()

    def initialize_weights(self):
        # 通用初始化
        self.apply(self._init_weights)
        
        # 🌟 关键点 4: AdaLN 零初始化 (Zero Init)
        # 让所有 AdaLN 的 gate 和 shift/scale 初始为 0
        # 这意味着初始状态下，AR Head 相当于一个 Identity 映射，不受 Backbone 干扰
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, z, targets=None):
        """
        z: [B, L, H]
        targets: [B, L, A]
        """
        B, L, H = z.shape
        A = self.augment_ratio
        
        # 1. 准备 Condition (AdaLN)
        z_flat = z.reshape(B * L, H) 
        cond = self.condition_proj(z_flat) 
        cond = cond.unsqueeze(1) # [B*L, 1, H]
        cond = cond + self.cond_pos_emb
        
        # 2. 准备 Input
        if targets is not None:
            t_flat = targets.reshape(B * L, A)
            target_emb = self.input_emb(t_flat)
            sos_emb = self.sos_token.expand(B * L, -1, -1)
            x = torch.cat([sos_emb, target_emb[:, :-1, :]], dim=1) # [B*L, A, H]
        else:
            x = self.sos_token.expand(B * L, A, -1)
        
        x = x + self.pos_emb 
        
        # 3. Mask
        mask = torch.empty(A, A, device=z.device).fill_(float("-inf")).triu_(1)
        
        # 4. Blocks
        for block in self.blocks:
            x = block(x, c=cond, attn_mask=mask)
            
        # 5. Final Layer
        x = self.final_layer(x, c=cond)
        
        # 6. Heads
        logits_list = []
        for i in range(A):
            step_logits = self.heads[i](x[:, i, :]) 
            logits_list.append(step_logits)
            
        logits = torch.stack(logits_list, dim=1) # [B*L, A, V]
        
        # 还原形状: [B, L, V, A]
        return logits.view(B, L, A, -1).permute(0, 1, 3, 2)

    @torch.no_grad()
    def sample(self, cond_z, uncond_z=None, cfg_scale=1.0, temperature=1.0):
        """
        带有 Logits 层面 CFG 的推理采样
        """
        B, L, H = cond_z.shape
        A = self.augment_ratio
        
        # 1. 准备条件 (Condition)
        cond_flat = cond_z.reshape(B * L, H)
        c_cond = self.condition_proj(cond_flat).unsqueeze(1) + self.cond_pos_emb
        
        do_cfg = False
        if uncond_z is not None and cfg_scale != 0.0 and cfg_scale != 1.0:
            do_cfg = True
            uncond_flat = uncond_z.reshape(B * L, H)
            c_uncond = self.condition_proj(uncond_flat).unsqueeze(1) + self.cond_pos_emb
            
        curr_input = self.sos_token.expand(B * L, 1, H)
        generated_ids = []
        all_log_probs = []
        input_embs_list = [curr_input] 

        for i in range(A):
            x = torch.cat(input_embs_list, dim=1) # [B*L, i+1, H]
            x = x + self.pos_emb[:, :i+1, :]
            mask = torch.empty(i+1, i+1, device=cond_z.device).fill_(float("-inf")).triu_(1)
            
            # 2. CFG 并行计算
            if do_cfg:
                # 在 Batch 维度拼接，并行计算 Cond 和 Uncond，节省时间
                x_double = torch.cat([x, x], dim=0) # [2*B*L, i+1, H]
                c_double = torch.cat([c_cond[:, :i+1, :], c_uncond[:, :i+1, :]], dim=0)
                
                out = x_double
                for block in self.blocks:
                    out = block(out, c=c_double, attn_mask=mask)
                out = self.final_layer(out, c=c_double)
                
                logits_double = self.heads[i](out[:, -1, :]) # [2*B*L, V]
                cond_logits, uncond_logits = logits_double.chunk(2, dim=0)
                
                # 🔥🔥🔥 【核心修正】在 Logits 层面执行 CFG 🔥🔥🔥
                logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            else:
                # 无 CFG 或 scale=1 时的正常逻辑
                out = x
                current_cond = c_cond[:, :i+1, :]
                for block in self.blocks:
                    out = block(out, c=current_cond, attn_mask=mask)
                out = self.final_layer(out, c=current_cond)
                logits = self.heads[i](out[:, -1, :]) 
            
            # 3. 采样逻辑 (自带温度保护)
            base_probs = F.softmax(logits, dim=-1)
            
            if temperature < 1e-6:
                next_token = torch.argmax(logits, dim=-1)
            else:
                scaled_probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(scaled_probs, num_samples=1).squeeze(-1)
            
            generated_ids.append(next_token)
            
            # 记录 Confidence (基于基础概率)
            selected_probs = torch.gather(base_probs, 1, next_token.unsqueeze(1)).squeeze(1)
            all_log_probs.append(torch.log(selected_probs + 1e-10))
            
            if i < A - 1:
                next_emb = self.input_emb(next_token).unsqueeze(1)
                input_embs_list.append(next_emb)
        
        ids = torch.stack(generated_ids, dim=1).view(B, L, A)
        avg_log_prob = torch.stack(all_log_probs, dim=1).mean(dim=1).view(B, L)
        confidence = torch.exp(avg_log_prob)
        
        return ids, confidence

        
class ImageBert(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-generation"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        self.target_codebook_size = config.model.vq_model.codebook_size
        self.condition_num_classes = config.model.generator.condition_num_classes
        self.image_seq_len = config.model.generator.image_seq_len
        self.mask_token_id = self.target_codebook_size
        self.hidden_size = config.model.generator.hidden_size
        self.num_hidden_layers = config.model.generator.num_hidden_layers
        self.num_attention_heads = config.model.generator.num_attention_heads
        self.intermediate_size = config.model.generator.intermediate_size
        self.augment_ratio = config.model.vq_model.get("augment_ratio", 1)
        self.feature_norm = nn.LayerNorm(self.hidden_size, eps=1e-6, elementwise_affine=False)
        bert_config = BertConfig(
            vocab_size=1, # We handle vocab size manually
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act='gelu',
            hidden_dropout_prob=config.model.generator.dropout,
            attention_probs_dropout_prob=config.model.generator.attn_drop,
            max_position_embeddings=config.model.generator.image_seq_len + 1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
        )
        # Ensure HF BERT uses a valid attention implementation on this env
        attn_impl = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
        setattr(bert_config, "_attn_implementation", attn_impl)
        setattr(bert_config, "attn_implementation", attn_impl)

        # --- Input Embeddings --- 
        if self.augment_ratio > 1:
            assert self.hidden_size % self.augment_ratio == 0, "hidden_size must be divisible by augment_ratio"
            sub_embed_dim = self.hidden_size // self.augment_ratio
            # 申请一个大的 Embedding，大小是 (A * V, D_sub)
            self.image_token_embeddings = nn.Embedding(
                self.augment_ratio * (self.target_codebook_size + 1), 
                sub_embed_dim
            )
            # 注册偏移量 buffer，会自动随模型保存和移动设备
            # 偏移量是 [0, 1025, 2050, 3075...]
            offsets = torch.arange(self.augment_ratio) * (self.target_codebook_size + 1)
            self.register_buffer('offsets', offsets)

        else:
            self.image_token_embeddings = nn.ModuleList([
                nn.Embedding(self.target_codebook_size + 1, self.hidden_size)
            ])

        self.condition_embedding = nn.Embedding(self.condition_num_classes + 2, self.hidden_size)
        self.position_embeddings = BertEmbeddings(bert_config).position_embeddings
        self.LayerNorm = BertEmbeddings(bert_config).LayerNorm
        self.dropout = BertEmbeddings(bert_config).dropout

        # --- Transformer Body ---
        self.transformer = BertEncoder(bert_config)

        # --- [修复 2] Output LM Heads (这里才是 LocalARHead) ---
        if self.augment_ratio > 1:
            # 使用 LocalARHead
            self.lm_head = LocalARHead(
                hidden_size=self.hidden_size, 
                codebook_size=self.target_codebook_size, 
                augment_ratio=self.augment_ratio,
                num_layers=2 # 推荐 2-4 层
            )
        else:
            self.lm_head = nn.Linear(self.hidden_size, self.target_codebook_size, bias=True)
        
        if hasattr(self, "post_init"):
            self.post_init()

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        dict_config = OmegaConf.to_container(self.config)
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)
    
    def get_backbone_features(self, input_ids, condition, cond_drop_prob=0.1):
        """
        辅助函数：只运行 Backbone 提取特征 z，不运行 Head。
        逻辑复用自原 forward 的前半部分。
        """
        # 1. Prepare condition token
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        # Use a dedicated uncond id within condition embedding range
        cond_ids = condition.clone()
        uncond_id = self.condition_num_classes  # reserve index C as uncond
        cond_ids[drop_label_mask] = uncond_id
        condition_embeds = self.condition_embedding(cond_ids).unsqueeze(1) # [B, 1, H]

        # 2. Prepare image tokens
        if input_ids is None:
            raise NotImplementedError

        if self.augment_ratio > 1:
            # 偏移查表
            # input_ids shape: [B, L, A]
            # self.offsets shape: [A]
            # 利用广播机制直接相加: [B, L, A]
            shifted_ids = input_ids + self.offsets
            # 一次性查表: [B, L, A, D_sub]
            image_embeds = self.image_token_embeddings(shifted_ids)
            # 展平最后两维: [B, L, A * D_sub] -> [B, L, D_full]
            B, L, A, _ = image_embeds.shape
            image_embeds = image_embeds.view(B, L, -1)
        else:
            if isinstance(self.image_token_embeddings, nn.ModuleList):
                image_embeds = self.image_token_embeddings[0](input_ids)
            else:
                image_embeds = self.image_token_embeddings(input_ids)

        # 3. Concatenate and add position embeddings
        final_embeddings = torch.cat([condition_embeds, image_embeds], dim=1)
        position_ids = torch.arange(final_embeddings.size(1), dtype=torch.long, device=final_embeddings.device).expand((final_embeddings.size(0), -1))
        final_embeddings += self.position_embeddings(position_ids)
        final_embeddings = self.LayerNorm(final_embeddings)
        final_embeddings = self.dropout(final_embeddings)

        # 4. Pass through Transformer
        transformer_output = self.transformer(final_embeddings)[0] 
        
        # 返回 z [B, L, H] (去掉 condition token)
        return transformer_output[:, 1:]
    
    def forward(self, input_ids=None, condition=None, cond_drop_prob=0.1, targets=None):
        
        # 1. 获取 Backbone 特征 (Masked Input -> z)
        image_output = self.get_backbone_features(input_ids, condition, cond_drop_prob)
        
        # [关键修复 1] Feature Normalization
        # 必须确保这一行存在！它负责把 Backbone 输出的数值拉回标准范围
        image_output = self.feature_norm(image_output)

        # 2. 计算 Logits (进入 AR Head)
        if self.augment_ratio > 1:
            # 训练时：targets 用于 Teacher Forcing
            # 推理时：targets 为 None，用 input_ids (Masked) 兜底
            if targets is None:
                targets = input_ids
            
            # AR Head 接收特征，计算 Loss
            final_logits = self.lm_head(image_output, targets=targets)
        else:
            # 兼容旧逻辑
            logits = self.lm_head(image_output)
            final_logits = logits.unsqueeze(-1)

        return final_logits
    
    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale=3.0,
                 guidance_decay="constant",
                 guidance_scale_pow=3.0,
                 randomize_temperature=4.5,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8):
        
        if guidance_decay not in ["constant", "linear", "power-cosine"]:
            raise ValueError(f"Unsupported guidance decay {guidance_decay}")
        
        device = condition.device
        ids = torch.full((condition.shape[0], self.image_seq_len, self.augment_ratio),
                          self.mask_token_id, device=device)
        
        cfg_scale = guidance_scale if guidance_decay == "constant" else 0.

        for step in range(num_sample_steps):
            ratio = 1. * (step + 1) / num_sample_steps
            annealed_temp = randomize_temperature * (1.0 - ratio)
            
            # 判断 mask
            is_mask = (ids == self.mask_token_id).any(dim=-1)

# --- 核心生成逻辑开始 ---
            
            # 1. 独立提取并 Normalize 特征
            cond_z = self.get_backbone_features(ids, condition, cond_drop_prob=0.0)
            cond_z = self.feature_norm(cond_z) # 必须保持与训练分布一致
            
            if cfg_scale != 0 and cfg_scale != 1.0:
                uncond_z = self.get_backbone_features(ids, condition, cond_drop_prob=1.0)
                uncond_z = self.feature_norm(uncond_z) 
            else:
                uncond_z = None

            # 2. 调用 Head 生成
            if self.augment_ratio > 1:
                # 把 CFG 的任务交给 AR Head 内部去处理
                sampled_ids, confidence_score = self.lm_head.sample(
                    cond_z=cond_z, 
                    uncond_z=uncond_z, 
                    cfg_scale=cfg_scale, 
                    temperature=annealed_temp
                )
                
                # Confidence 处理
                confidence = torch.where(is_mask, confidence_score, torch.full_like(confidence_score, float('inf')))

            else:
                # 兼容不带 AR 的旧逻辑
                if uncond_z is not None:
                    z = uncond_z + (cond_z - uncond_z) * cfg_scale
                else:
                    z = cond_z
                logits = self.lm_head(z).unsqueeze(-1)
                
                # 🔥 补上漏掉的原版采样和 Confidence 计算逻辑 🔥
                if softmax_temperature_annealing:
                    logits = logits / (0.5 + 0.8 * (1 - ratio))

                def log(t, eps=1e-20): return torch.log(t.clamp(min=eps))
                def gumbel_noise(t): return -log(-log(torch.zeros_like(t).uniform_(0, 1)))
                
                noisy_logits = logits + annealed_temp * gumbel_noise(logits)
                sampled_ids = noisy_logits.argmax(dim=-2) 
                
                probs = F.softmax(logits, dim=-2)
                sampled_probs = torch.gather(probs, dim=-2, index=sampled_ids.unsqueeze(-2)).squeeze(-2)
                confidence = sampled_probs.squeeze(-1)
                confidence = torch.where(is_mask, confidence, torch.full_like(confidence, float('inf')))  
            # --- 核心生成逻辑结束 ---
            
            # Apply Mask Replacement
            is_mask_expanded = is_mask.unsqueeze(-1).expand_as(ids)
            sampled_ids = torch.where(is_mask_expanded, sampled_ids, ids)

            # Re-masking Logic
            if guidance_decay == "power-cosine":
                 guidance_scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                 scale_step = (1 - torch.cos(((step / num_sample_steps) ** guidance_scale_pow) * torch.pi)) * 1/2
                 cfg_scale = (guidance_scale - 1) * scale_step + 1

            mask_ratio = np.arccos(ratio) / (math.pi * 0.5)
            mask_len = torch.Tensor([np.floor(self.image_seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                     torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1,
                                                   mask_len))[0].squeeze()
            
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
            masking = (confidence <= cut_off)
            
            if step == num_sample_steps - 1:
                ids = sampled_ids
            else:
                masking_expanded = masking.unsqueeze(-1).expand_as(ids)
                ids = torch.where(masking_expanded, self.mask_token_id, sampled_ids)

            if guidance_decay == "linear":
                cfg_scale = ratio * guidance_scale
                
        return ids

    def masking_input_tokens(self, input_tokens):
        batch_size, seq_len, _ = input_tokens.shape
        device = input_tokens.device

        timesteps = torch.zeros((batch_size,), device=device).float().uniform_(0, 1.0)
        mask_ratio = torch.acos(timesteps) / (math.pi * 0.5) # arccos schedule
        mask_ratio = torch.clamp(mask_ratio, min=1e-6, max=1.)
        num_token_masked = (seq_len * mask_ratio).round().clamp(min=1)
        batch_randperm = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        masks = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        
        masks_expanded = masks.unsqueeze(-1).expand_as(input_tokens) 
        masked_tokens = torch.where(masks_expanded, self.mask_token_id, input_tokens)
        return masked_tokens, masks


class UViTBert(ImageBert):
    def __init__(self, config):
        # 调用父类初始化，这会正确创建 self.lm_head (LocalARHead) 以及 self.offsets
        super().__init__(config=config)

        # 删除父类的 BERT 本体
        del self.transformer
        del self.position_embeddings
        del self.LayerNorm
        del self.dropout

        # UViT 特有的位置编码
        self.pos_embed = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, self.config.model.generator.image_seq_len + 1, self.hidden_size)), 0., 0.02)
 
        self.in_blocks = nn.ModuleList([
            UViTBlock(
                dim=self.hidden_size, num_heads=self.num_attention_heads, mlp_ratio=(self.intermediate_size / self.hidden_size),
                qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, use_checkpoint=False)
            for _ in range(self.num_hidden_layers // 2)])

        self.mid_block = UViTBlock(
                dim=self.hidden_size, num_heads=self.num_attention_heads, mlp_ratio=(self.intermediate_size / self.hidden_size),
                qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, use_checkpoint=False)

        self.out_blocks = nn.ModuleList([
            UViTBlock(
                dim=self.hidden_size, num_heads=self.num_attention_heads, mlp_ratio=(self.intermediate_size / self.hidden_size),
                qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, skip=True, use_checkpoint=False)
            for _ in range(self.num_hidden_layers // 2)])

        self.norm = nn.LayerNorm(self.hidden_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_backbone_features(self, input_ids, condition, cond_drop_prob=0.1):
        """
        UViT 版本的特征提取 Helper
        """
        # 1. Prepare condition
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        condition = condition + self.target_codebook_size + 1
        condition[drop_label_mask] = self.condition_num_classes + self.target_codebook_size + 1
        condition_embeds = self.condition_embedding(condition).unsqueeze(1)

        # 2. Prepare image tokens
        if self.augment_ratio > 1:
            shifted_ids = input_ids + self.offsets # self.offsets 来自父类
            image_embeds = self.image_token_embeddings(shifted_ids)
            B, L, A, _ = image_embeds.shape
            image_embeds = image_embeds.view(B, L, -1)
        else:
            if isinstance(self.image_token_embeddings, nn.ModuleList):
                image_embeds = self.image_token_embeddings[0](input_ids)
            else:
                image_embeds = self.image_token_embeddings(input_ids)

        # 3. Concatenate and Position Embedding
        final_embeddings = torch.cat([condition_embeds, image_embeds], dim=1)
        
        # UViT 特有的 pos_embed 加法
        x = final_embeddings + self.pos_embed[:, :final_embeddings.shape[1]]
        
        # 4. UViT Blocks Pass
        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)
        x = self.mid_block(x)
        for blk in self.out_blocks:
            x = blk(x, skips.pop())
        x = self.norm(x)
        
        # 返回 z (去掉 condition token)n
        return x[:, 1:]

    def forward(self, input_ids=None, condition=None, cond_drop_prob=0.1, targets=None): # <--- [修复1] 加上 targets
        # 1. 获取 Backbone 特征
        # Backbone 依然看 input_ids (Masked)
        image_output = self.get_backbone_features(input_ids, condition, cond_drop_prob)

        # Feature Normalization
        image_output = self.feature_norm(image_output)
        # 2. 获取 Logits
        if self.augment_ratio > 1:
            # [修复2] 兜底逻辑：如果没传 targets (推理时)，用 input_ids
            if targets is None:
                targets = input_ids
                
            # [修复3] 传入正确的 targets (Clean)
            final_logits = self.lm_head(image_output, targets=targets)
        else:
            final_logits = self.lm_head(image_output).unsqueeze(-1)

        return final_logits