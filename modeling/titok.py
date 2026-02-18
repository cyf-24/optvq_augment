"""This file contains the model definition of TiTok.

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
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from timm.models import create_model

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
from modeling.quantizer.optvq import OptVQ as SinkhornVectorQuantizer
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
from modeling.modules.clip_loss import ClipLoss
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin

class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return x * self.std + self.mean

class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)


class TiTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-tokenization"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        self.augment_ratio = config.model.vq_model.augment_ratio
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae", "optvq"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq", "optvq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,)
        elif self.quantize_mode == "optvq":
            self.quantize = SinkhornVectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size // config.model.vq_model.augment_ratio,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,
                use_shared_linear=True,
                use_sinkhorn=True,
                num_group=1)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError

        self.semantic_guide = None
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
            
            # semantic guidance
            self.semantic_guide = config.model.vq_model.get("semantic_guide", None)
            if self.semantic_guide == "dinov2":
                # build semantic model
                semantic_model = create_model(
                    model_name="vit_base_patch14_dinov2.lvd142m",
                    pretrained=True, img_size=256, patch_size=16, 
                    drop_path_rate=0.0
                )
                semantic_model.eval()
                for param in semantic_model.parameters():
                    param.requires_grad = False
                self.semantic_model = semantic_model
                # build semantic loss
                self.semantic_loss = ClipLoss(
                    local_loss=True,
                    gather_with_grad=True,
                    cache_labels=True,
                    rank=dist.get_rank(),
                    world_size=dist.get_world_size()
                )
                self.sem_loss_weight = config.model.vq_model.get("sem_loss_weight", 0.0)
                # build semantic layer

                head_dim = self.pixel_quantize.embedding.weight.size(-1)
                embed_dim = head_dim * self.augment_ratio

                self.sem_norm = nn.LayerNorm(embed_dim, eps=1e-6)
                self.sem_linear = nn.Linear(embed_dim, 768)
                self.sem_scale = 1 # nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                
                # sem_normalize
                self.sem_denormalize = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                self.sem_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
                # result_dict["quantizer_loss"] *= 0
                # result_dict["commitment_loss"] *= 0
                # result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            if self.quantize_mode == "vq" or self.quantize_mode == "optvq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        return z_quantized, result_dict
    
    def decode(self, z_quantized, return_latent: bool = False):
        decoded = self.decoder(z_quantized)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded = self.pixel_decoder(quantized_states)
            # 返回的是真实的图像
            if return_latent:
                return dict(
                    decoded=decoded,
                    latent=quantized_states
                )
            else:
                return decoded
        return decoded
    
    def decode_tokens(self, tokens):
        # 期望输入形状: [B, L, A]，A=augment_ratio，多路分组码本一一对应
        assert self.quantize_mode == "optvq", "decode_tokens 仅支持 optvq 多路分组码本"
        assert tokens.dim() == 3, f"tokens 应为 [B, L, A]，当前 {tokens.shape}"
        B, L, A = tokens.shape
        assert hasattr(self.quantize, "num_group") and self.quantize.num_group * self.augment_ratio == A, \
            f"分组数不匹配: num_group={getattr(self.quantize, 'num_group', None)} * self.augment_ratio={self.augment_ratio} vs A={A}"

        indices = tokens.reshape(-1, A)  # (B*L, A)
        zq_flat = self.quantize.get_codebook_entry(indices)  # (B*L*A, D_sub)
        D_sub = zq_flat.shape[-1]
        zq = zq_flat.view(B, L, A, D_sub).reshape(B, L, A * D_sub)
        zq = rearrange(zq, 'b l c -> b c 1 l').contiguous()
        return self.decode(zq)
    
    def forward(self, x, use_semantic: bool = False):
        z_quantized, result_dict = self.encode(x)
        out = self.decode(z_quantized, return_latent=True)
        if isinstance(out, dict):
            decoded = out["decoded"]
            latent = out["latent"]
        else:
            decoded = out
            latent = None
        
        # semantic guidance
        if self.semantic_guide is not None and use_semantic:
            # x is in the range [0, 1]
            x_copy = self.sem_normalize(x)
            # compute the semantic reference
            with torch.no_grad():
                clip_ref = self.semantic_model(x_copy)
                clip_ref = F.normalize(clip_ref, dim=-1, p=2)
            # compute the projected latent
            clip_vis = torch.mean(latent, dim=(2, 3))
            clip_vis = self.sem_norm(clip_vis)
            clip_vis = self.sem_linear(clip_vis)
            clip_vis = F.normalize(clip_vis, dim=-1, p=2)

            with torch.amp.autocast('cuda', enabled=False):
                sem_loss = self.semantic_loss(
                    image_features=clip_vis.float(),
                    text_features=clip_ref.float(),
                    logit_scale=self.sem_scale.exp(),
                ) * self.sem_loss_weight
                result_dict["sem_loss"] = sem_loss
                result_dict["quantizer_loss"] = result_dict["quantizer_loss"] + sem_loss
        else:
            result_dict["sem_loss"] = torch.tensor(0.0, device=decoded.device)

        return decoded, result_dict