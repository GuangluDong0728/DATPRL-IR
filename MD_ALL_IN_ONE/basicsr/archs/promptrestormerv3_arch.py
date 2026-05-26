import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== Reuse building blocks from your existing Restormer implementation ====
from .restormer_arch import (
    OverlapPatchEmbed, TransformerBlock, Downsample, Upsample, LayerNorm
)

# ==== Reuse your dual-prompt pools and feature mappers (already in your repo) ====
from .promptpoolnafnetv3_arch import (
    DomainPromptPool, DegradationPromptPool,
    DomainFeatureMapper, DegradationFeatureMapper
)


# -------------------------------------------------------------
# Prompt-enhanced Transformer block (Restormer-style)
# -------------------------------------------------------------
class PromptEnhancedTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias', use_prompt=False, prompt_dim=768,
                 fusion_method='cross_attention'):
        super().__init__()
        self.use_prompt = use_prompt
        self.fusion_method = fusion_method
        self.dim = dim

        # Restormer submodules
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn  = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type).attn
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type).ffn

        # ===== Learnable feature/prompt weights (same spirit as NAF version) =====
        if use_prompt:
            self.feature_weight = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
            self.prompt_weight  = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
            self.weight_norm    = nn.Softmax(dim=0)

            if fusion_method == 'concat':
                # project prompt to feature dim then fuse by 1x1 conv after concat
                self.prompt_projection = nn.Linear(prompt_dim, dim)
                self.prompt_fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)
            elif fusion_method == 'cross_attention':
                # single-token prompt attends into spatial features
                self.cross_attn_q = nn.Linear(dim, dim)
                self.cross_attn_k = nn.Linear(prompt_dim, dim)
                self.cross_attn_v = nn.Linear(prompt_dim, dim)
                self.cross_attn_out = nn.Linear(dim, dim)
                self.prompt_norm = nn.LayerNorm(prompt_dim)
            elif fusion_method == 'adaptive_modulation':
                # FiLM-like modulation
                self.prompt_mlp = nn.Sequential(
                    nn.Linear(prompt_dim, dim * 2),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim * 2)
                )

    def get_normalized_weights(self):
        weights = torch.stack([self.feature_weight, self.prompt_weight])
        normalized = self.weight_norm(weights)
        return normalized[0], normalized[1]

    def apply_prompt_fusion(self, x, combined_prompt):
        if not self.use_prompt or combined_prompt is None:
            return x
        B, C, H, W = x.shape
        feat_w, pr_w = self.get_normalized_weights()

        if self.fusion_method == 'concat':
            weighted_x = x * feat_w
            proj = self.prompt_projection(combined_prompt)  # [B, C]
            proj = proj.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
            weighted_p = proj * pr_w
            x = self.prompt_fusion(torch.cat([weighted_x, weighted_p], dim=1))

        elif self.fusion_method == 'cross_attention':
            x_flat = x.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
            prompt_seq = self.prompt_norm(combined_prompt.unsqueeze(1))  # [B, 1, P]
            q = self.cross_attn_q(x_flat) * feat_w      # [B, HW, C]
            k = self.cross_attn_k(prompt_seq) * pr_w    # [B, 1, C]
            v = self.cross_attn_v(prompt_seq) * pr_w    # [B, 1, C]
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)  # [B, HW, 1]
            attn = torch.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn, v)  # [B, HW, C]
            out = self.cross_attn_out(out)
            x = x * feat_w + out.transpose(1, 2).reshape(B, C, H, W)

        elif self.fusion_method == 'adaptive_modulation':
            mod = self.prompt_mlp(combined_prompt)  # [B, 2C]
            scale, shift = mod.chunk(2, dim=1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            x = x * feat_w * (1 + scale * pr_w) + shift * pr_w

        return x

    def forward(self, x, combined_prompt=None):
        # Fuse prompt before first norm/attention block
        x_in = self.apply_prompt_fusion(x, combined_prompt)
        x = x + self.attn(self.norm1(x_in))
        x = x + self.ffn(self.norm2(x))
        return x


# -------------------------------------------------------------
# Dual-Prompt Restormer v3
# -------------------------------------------------------------
@ARCH_REGISTRY.register()
class DualPromptRestormerv3(nn.Module):

    def __init__(self,
                 img_channel=3,
                 # Restormer base width (a.k.a. dim)
                 dim=None,
                 width=None,  
                 enc_blk_nums=[4, 6, 6],   # encoder levels 1..3
                 middle_blk_num=8,        # latent level (level 4)
                 dec_blk_nums=[6, 6, 4],  # decoder levels 1..3
                 refine_blk_num=4,        # refinement blocks after decoder level1
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 # Prompt system
                 use_prompt=True,
                 domain_pool_size=15,
                 degradation_pool_size=15,
                 domain_prompt_dim=1024,
                 degradation_prompt_dim=1024,
                 domain_top_k=5,
                 degradation_top_k=3,
                 value_num_tokens=2,
                 prompt_fusion_method='cross_attention',
                 fusion_method='cross_attention'):
        super().__init__()

        self.use_prompt = use_prompt
        self.fusion_method = fusion_method
        self.prompt_fusion_method = prompt_fusion_method

        if dim is None:
            if width is None:
                dim = 48
            else:
                dim = width

        # ----------------------------
        # Stems
        # ----------------------------
        self.patch_embed = OverlapPatchEmbed(img_channel, dim)
        self.output = nn.Conv2d(int(dim * 2**1), img_channel, kernel_size=3, stride=1, padding=1, bias=bias)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(enc_blk_nums[0])
        ])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(enc_blk_nums[1])
        ])
        self.down2_3 = Downsample(int(dim*2**1))

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(enc_blk_nums[2])
        ])
        self.down3_4 = Downsample(int(dim*2**2))

        # ----------------------------
        # Latent (prompt-enhanced)
        # ----------------------------
        self.latent_blocks = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**3), num_heads=heads[3],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=(domain_prompt_dim if prompt_fusion_method=='cross_attention' else (domain_prompt_dim+degradation_prompt_dim)),
                                            fusion_method=fusion_method)
            for _ in range(middle_blk_num)
        ])

        # ----------------------------
        # Decoder (prompt-enhanced)
        # ----------------------------
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=(domain_prompt_dim if prompt_fusion_method=='cross_attention' else (domain_prompt_dim+degradation_prompt_dim)),
                                            fusion_method=fusion_method)
            for _ in range(dec_blk_nums[2])
        ])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**1), num_heads=heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=(domain_prompt_dim if prompt_fusion_method=='cross_attention' else (domain_prompt_dim+degradation_prompt_dim)),
                                            fusion_method=fusion_method)
            for _ in range(dec_blk_nums[1])
        ])

        self.up2_1 = Upsample(int(dim*2**1))
        # (Restormer decoder level1 concatenates with level1 encoder; no 1x1 before blocks)
        self.decoder_level1 = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**1), num_heads=heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=(domain_prompt_dim if prompt_fusion_method=='cross_attention' else (domain_prompt_dim+degradation_prompt_dim)),
                                            fusion_method=fusion_method)
            for _ in range(dec_blk_nums[0])
        ])

        # refinement blocks (prompt-enhanced)
        self.refinement = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**1), num_heads=heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=(domain_prompt_dim if prompt_fusion_method=='cross_attention' else (domain_prompt_dim+degradation_prompt_dim)),
                                            fusion_method=fusion_method)
            for _ in range(refine_blk_num)
        ])

        # ----------------------------
        # Prompt pools + feature mappers
        # ----------------------------
        if use_prompt:
            self.domain_prompt_pool = DomainPromptPool(
                pool_size=domain_pool_size,
                prompt_dim=domain_prompt_dim,
                top_k=domain_top_k,
                value_num_tokens=value_num_tokens
            )
            self.degradation_prompt_pool = DegradationPromptPool(
                pool_size=degradation_pool_size,
                prompt_dim=degradation_prompt_dim,
                top_k=degradation_top_k,
                value_num_tokens=value_num_tokens
            )
            # mapper inputs: first-layer channels = dim; deepest encoder channels = dim*8
            self.domain_mapper      = DomainFeatureMapper(dim, domain_prompt_dim)
            self.degradation_mapper = DegradationFeatureMapper(int(dim*2**3), degradation_prompt_dim)

            # prompt cross-attention modules for fusing domain & degradation prompts
            if prompt_fusion_method == 'cross_attention':
                self.prompt_cross_attention = nn.MultiheadAttention(
                    embed_dim=domain_prompt_dim,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                )
                self.prompt_fusion_norm = nn.LayerNorm(domain_prompt_dim)
                self.prompt_fusion_ffn  = nn.Sequential(
                    nn.Linear(domain_prompt_dim, domain_prompt_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(domain_prompt_dim * 2, domain_prompt_dim)
                )

        self.padder_size = 8  

    def intro(self, x):
        return self.patch_embed(x)

    def fuse_prompts(self, domain_prompt, degradation_prompt):
        if self.prompt_fusion_method == 'concat':
            return torch.cat([domain_prompt, degradation_prompt], dim=1)
        elif self.prompt_fusion_method == 'cross_attention':
            # use domain as query, degradation as key/value (pre-created modules)
            q = domain_prompt.unsqueeze(1)
            k = degradation_prompt.unsqueeze(1)
            v = degradation_prompt.unsqueeze(1)
            att, _ = self.prompt_cross_attention(q, k, v)
            out = self.prompt_fusion_norm(att + q)
            out = out + self.prompt_fusion_ffn(out)
            return out.squeeze(1)
        else:
            raise ValueError(f"Unknown prompt fusion method: {self.prompt_fusion_method}")

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape
        x_in = self.check_image_size(inp_img)

        # Encoder
        enc1 = self.patch_embed(x_in)
        out1 = self.encoder_level1(enc1)
        x = self.down1_2(out1)

        out2 = self.encoder_level2(x)
        x = self.down2_3(out2)

        out3 = self.encoder_level3(x)
        x = self.down3_4(out3)  # deepest encoder features
        encoded_features = x

        # Prompt selection & fusion
        combined_prompt = None
        if self.use_prompt:
            domain_query = self.domain_mapper(enc1)              # [B, Dd]
            domain_prompt, domain_indices = self.domain_prompt_pool(domain_query)

            degradation_query = self.degradation_mapper(encoded_features)  # [B, Dg]
            degradation_prompt, degradation_indices = self.degradation_prompt_pool(degradation_query)

            combined_prompt = self.fuse_prompts(domain_prompt, degradation_prompt)

        # Latent (prompt-enhanced)
        for blk in self.latent_blocks:
            x = blk(x, combined_prompt)

        # Decoder level3
        x = self.up4_3(x)
        x = torch.cat([x, out3], dim=1)
        x = self.reduce_chan_level3(x)
        for blk in self.decoder_level3:
            x = blk(x, combined_prompt)

        # Decoder level2
        x = self.up3_2(x)
        x = torch.cat([x, out2], dim=1)
        x = self.reduce_chan_level2(x)
        for blk in self.decoder_level2:
            x = blk(x, combined_prompt)

        # Decoder level1
        x = self.up2_1(x)
        x = torch.cat([x, out1], dim=1)
        for blk in self.decoder_level1:
            x = blk(x, combined_prompt)

        # Refinement
        for blk in self.refinement:
            x = blk(x, combined_prompt)

        out = self.output(x) + x_in
        return out[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

@ARCH_REGISTRY.register()
class DualPromptRestormerv3_only_task_prompt(nn.Module):

    def __init__(self,
                 img_channel=3,
                 # Restormer base width (a.k.a. dim)
                 dim=None,
                 width=None, 
                 enc_blk_nums=[4, 6, 6],   # encoder levels 1..3
                 middle_blk_num=8,        # latent level (level 4)
                 dec_blk_nums=[6, 6, 4],  # decoder levels 1..3
                 refine_blk_num=4,        # refinement blocks after decoder level1
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 # Prompt system
                 use_prompt=True,
                 degradation_pool_size=15,
                 degradation_prompt_dim=1024,
                 degradation_top_k=3,
                 value_num_tokens=2,
                 fusion_method='cross_attention'):
        super().__init__()

        self.use_prompt = use_prompt
        self.fusion_method = fusion_method

        if dim is None:
            if width is None:
                dim = 48
            else:
                dim = width

        # ----------------------------
        # Stems
        # ----------------------------
        self.patch_embed = OverlapPatchEmbed(img_channel, dim)
        self.output = nn.Conv2d(int(dim * 2**1), img_channel, kernel_size=3, stride=1, padding=1, bias=bias)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(enc_blk_nums[0])
        ])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(enc_blk_nums[1])
        ])
        self.down2_3 = Downsample(int(dim*2**1))

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(enc_blk_nums[2])
        ])
        self.down3_4 = Downsample(int(dim*2**2))

        # ----------------------------
        # Latent (prompt-enhanced)
        # ----------------------------
        self.latent_blocks = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**3), num_heads=heads[3],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=degradation_prompt_dim,
                                            fusion_method=fusion_method)
            for _ in range(middle_blk_num)
        ])

        # ----------------------------
        # Decoder (prompt-enhanced)
        # ----------------------------
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=degradation_prompt_dim,
                                            fusion_method=fusion_method)
            for _ in range(dec_blk_nums[2])
        ])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**1), num_heads=heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=degradation_prompt_dim,
                                            fusion_method=fusion_method)
            for _ in range(dec_blk_nums[1])
        ])

        self.up2_1 = Upsample(int(dim*2**1))
        
        self.decoder_level1 = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**1), num_heads=heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=degradation_prompt_dim,
                                            fusion_method=fusion_method)
            for _ in range(dec_blk_nums[0])
        ])

        # refinement blocks (prompt-enhanced)
        self.refinement = nn.ModuleList([
            PromptEnhancedTransformerBlock(dim=int(dim*2**1), num_heads=heads[0],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type,
                                            use_prompt=use_prompt, prompt_dim=degradation_prompt_dim,
                                            fusion_method=fusion_method)
            for _ in range(refine_blk_num)
        ])

        # ----------------------------
        # Prompt pools + feature mappers
        # ----------------------------
        if use_prompt:
            self.degradation_prompt_pool = DegradationPromptPool(
                pool_size=degradation_pool_size,
                prompt_dim=degradation_prompt_dim,
                top_k=degradation_top_k,
                value_num_tokens=value_num_tokens
            )
           
            self.degradation_mapper = DegradationFeatureMapper(int(dim*2**3), degradation_prompt_dim)

        self.padder_size = 8  

    def intro(self, x):
        return self.patch_embed(x)

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape
        x_in = self.check_image_size(inp_img)

        # Encoder
        enc1 = self.patch_embed(x_in)
        out1 = self.encoder_level1(enc1)
        x = self.down1_2(out1)

        out2 = self.encoder_level2(x)
        x = self.down2_3(out2)

        out3 = self.encoder_level3(x)
        x = self.down3_4(out3)  # deepest encoder features
        encoded_features = x

        # Prompt selection & fusion
        combined_prompt = None
        if self.use_prompt:
            degradation_query = self.degradation_mapper(encoded_features)  # [B, Dg]
            degradation_prompt, degradation_indices = self.degradation_prompt_pool(degradation_query)

            combined_prompt = degradation_prompt

        # Latent (prompt-enhanced)
        for blk in self.latent_blocks:
            x = blk(x, combined_prompt)

        # Decoder level3
        x = self.up4_3(x)
        x = torch.cat([x, out3], dim=1)
        x = self.reduce_chan_level3(x)
        for blk in self.decoder_level3:
            x = blk(x, combined_prompt)

        # Decoder level2
        x = self.up3_2(x)
        x = torch.cat([x, out2], dim=1)
        x = self.reduce_chan_level2(x)
        for blk in self.decoder_level2:
            x = blk(x, combined_prompt)

        # Decoder level1
        x = self.up2_1(x)
        x = torch.cat([x, out1], dim=1)
        for blk in self.decoder_level1:
            x = blk(x, combined_prompt)

        # Refinement
        for blk in self.refinement:
            x = blk(x, combined_prompt)

        out = self.output(x) + x_in
        return out[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x