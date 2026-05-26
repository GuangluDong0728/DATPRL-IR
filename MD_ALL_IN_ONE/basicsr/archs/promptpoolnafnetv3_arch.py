import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class DomainPromptPool(nn.Module):

    def __init__(self, pool_size=6, prompt_dim=1024, top_k=3, value_num_tokens=5):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_dim = prompt_dim
        self.top_k = top_k
        self.value_num_tokens = value_num_tokens  
        

        self.prompt_keys = nn.Parameter(torch.randn(pool_size, prompt_dim))

        self.prompt_values = nn.Parameter(torch.randn(pool_size, value_num_tokens, prompt_dim))

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.value_aggregation = nn.Sequential(
            nn.Linear(value_num_tokens * prompt_dim, prompt_dim),
            nn.LayerNorm(prompt_dim),
            nn.GELU()
        )

        self.init_prompts_with_diversity()
    
    def init_prompts_with_diversity(self):

        for i in range(self.pool_size):
            if i < self.pool_size // 2:
                nn.init.xavier_uniform_(self.prompt_keys[i:i+1])
                for j in range(self.value_num_tokens):
                    nn.init.xavier_uniform_(self.prompt_values[i:i+1, j:j+1])
            else:
                nn.init.orthogonal_(self.prompt_keys[i:i+1])
                for j in range(self.value_num_tokens):
                    nn.init.orthogonal_(self.prompt_values[i:i+1, j:j+1])

        with torch.no_grad():
            for i in range(1, self.pool_size):
                self.prompt_keys[i] += 0.5 * torch.randn_like(self.prompt_keys[i])
                self.prompt_values[i] += 0.5 * torch.randn_like(self.prompt_values[i])
    
    def forward_with_temperature(self, query_feature, use_temperature=True):
        B = query_feature.shape[0]
        
        query_norm = F.normalize(query_feature, dim=1)
        keys_norm = F.normalize(self.prompt_keys, dim=1)

        similarity = torch.matmul(query_norm, keys_norm.t())  # [B, pool_size]
        
        if use_temperature:
            temperature = torch.clamp(self.temperature, min=0.1, max=2.0)
            similarity = similarity / temperature

        if not self.training or not use_temperature:
            _, top_indices = torch.topk(similarity, k=self.top_k, dim=1)
        else:

            noise = 0.1 * torch.randn_like(similarity)
            noisy_similarity = similarity + noise
            _, top_indices = torch.topk(noisy_similarity, k=self.top_k, dim=1)

        selected_values = self.prompt_values[top_indices]  # [B, top_k, value_num_tokens, prompt_dim]

        selected_similarities = torch.gather(similarity, 1, top_indices)  # [B, top_k]
        attention_weights = F.softmax(selected_similarities, dim=1)  # [B, top_k]

        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, top_k, 1, 1]
        weighted_values = selected_values * attention_weights  # [B, top_k, value_num_tokens, prompt_dim]
        aggregated_values = torch.sum(weighted_values, dim=1)  # [B, value_num_tokens, prompt_dim]

        B, num_tokens, dim = aggregated_values.shape
        aggregated_values_flat = aggregated_values.view(B, -1)  # [B, value_num_tokens * prompt_dim]
        domain_prompt = self.value_aggregation(aggregated_values_flat)  # [B, prompt_dim]
        
        return domain_prompt, top_indices
    
    def forward(self, query_feature):
        return self.forward_with_temperature(query_feature, use_temperature=True)


class DegradationPromptPool(nn.Module):
    def __init__(self, pool_size=15, prompt_dim=1024, top_k=3, value_num_tokens=5):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_dim = prompt_dim
        self.top_k = top_k
        self.value_num_tokens = value_num_tokens 

        self.prompt_keys = nn.Parameter(torch.randn(pool_size, prompt_dim))

        self.prompt_values = nn.Parameter(torch.randn(pool_size, value_num_tokens, prompt_dim))

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.value_aggregation = nn.Sequential(
            nn.Linear(value_num_tokens * prompt_dim, prompt_dim),
            nn.LayerNorm(prompt_dim),
            nn.GELU()
        )

        self.init_prompts_with_diversity()
    
    def init_prompts_with_diversity(self):

        for i in range(self.pool_size):
            if i < self.pool_size // 2:
                nn.init.xavier_uniform_(self.prompt_keys[i:i+1])
                for j in range(self.value_num_tokens):
                    nn.init.xavier_uniform_(self.prompt_values[i:i+1, j:j+1])
            else:
                nn.init.orthogonal_(self.prompt_keys[i:i+1])
                for j in range(self.value_num_tokens):
                    nn.init.orthogonal_(self.prompt_values[i:i+1, j:j+1])
                
        with torch.no_grad():
            for i in range(1, self.pool_size):
                self.prompt_keys[i] += 0.5 * torch.randn_like(self.prompt_keys[i])
                self.prompt_values[i] += 0.5 * torch.randn_like(self.prompt_values[i])
    
    def forward_with_temperature(self, query_feature, use_temperature=True):
        B = query_feature.shape[0]
        
        query_norm = F.normalize(query_feature, dim=1)
        keys_norm = F.normalize(self.prompt_keys, dim=1)

        similarity = torch.matmul(query_norm, keys_norm.t())  # [B, pool_size]
        
        if use_temperature:

            temperature = torch.clamp(self.temperature, min=0.1, max=2.0)
            similarity = similarity / temperature

        if not self.training or not use_temperature:
            _, top_indices = torch.topk(similarity, k=self.top_k, dim=1)
        else:
            noise = 0.1 * torch.randn_like(similarity)
            noisy_similarity = similarity + noise
            _, top_indices = torch.topk(noisy_similarity, k=self.top_k, dim=1)

        selected_values = self.prompt_values[top_indices]  # [B, top_k, value_num_tokens, prompt_dim]

        selected_similarities = torch.gather(similarity, 1, top_indices)  # [B, top_k]
        attention_weights = F.softmax(selected_similarities, dim=1)  # [B, top_k]

        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, top_k, 1, 1]
        weighted_values = selected_values * attention_weights  # [B, top_k, value_num_tokens, prompt_dim]
        aggregated_values = torch.sum(weighted_values, dim=1)  # [B, value_num_tokens, prompt_dim]

        B, num_tokens, dim = aggregated_values.shape
        aggregated_values_flat = aggregated_values.view(B, -1)  # [B, value_num_tokens * prompt_dim]
        degradation_prompt = self.value_aggregation(aggregated_values_flat)  # [B, prompt_dim]
        # print(degradation_prompt.shape)
        
        return degradation_prompt, top_indices
    
    def forward(self, query_feature):
        return self.forward_with_temperature(query_feature, use_temperature=True)


class DomainFeatureMapper(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  
            nn.Conv2d(input_dim, input_dim // 2, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(input_dim // 2, input_dim // 4, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(input_dim // 4, output_dim)
        )
    
    def forward(self, x):
        return self.mapper(x)

class DegradationFeatureMapper(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  
            nn.Conv2d(input_dim, input_dim // 2, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(input_dim // 2, input_dim // 4, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim // 4, output_dim)
        )
    
    def forward(self, x):
        return self.mapper(x)

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)
        return y

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class PromptEnhancedNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., 
                 use_prompt=False, prompt_dim=768, fusion_method='cross_attention'):
        super().__init__()
        dw_channel = c * DW_Expand
        self.use_prompt = use_prompt
        self.fusion_method = fusion_method

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        

        if use_prompt:

            self.feature_weight = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
            self.prompt_weight = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

            self.weight_norm = nn.Softmax(dim=0)
            
            if fusion_method == 'concat':
                self.prompt_projection = nn.Linear(prompt_dim, c)  
                self.prompt_fusion = nn.Conv2d(c * 2, c, kernel_size=1, bias=True)  # 2*c -> c
            elif fusion_method == 'cross_attention':
                self.cross_attn_q = nn.Linear(c, c)
                self.cross_attn_k = nn.Linear(prompt_dim, c)
                self.cross_attn_v = nn.Linear(prompt_dim, c)
                self.cross_attn_out = nn.Linear(c, c)
                self.prompt_norm = nn.LayerNorm(prompt_dim)
            elif fusion_method == 'adaptive_modulation':
                self.prompt_mlp = nn.Sequential(
                    nn.Linear(prompt_dim, c * 2),
                    nn.ReLU(),
                    nn.Linear(c * 2, c * 2)  
                )

    def get_normalized_weights(self):
        weights = torch.stack([self.feature_weight, self.prompt_weight])
        normalized_weights = self.weight_norm(weights)
        return normalized_weights[0], normalized_weights[1]

    def apply_prompt_fusion(self, x, combined_prompt):
        if not self.use_prompt or combined_prompt is None:
            return x
            
        B, C, H, W = x.shape

        feat_weight, prompt_weight = self.get_normalized_weights()
        
        if self.fusion_method == 'concat':

            weighted_x = x * feat_weight

            projected_prompt = self.prompt_projection(combined_prompt)  # [B, C]
            prompt_spatial = projected_prompt.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)  # [B, C, H, W]
            weighted_prompt = prompt_spatial * prompt_weight

            concat_features = torch.cat([weighted_x, weighted_prompt], dim=1)  # [B, 2*C, H, W]
            x = self.prompt_fusion(concat_features)  # [B, 2*C, H, W] -> [B, C, H, W]
            
        elif self.fusion_method == 'cross_attention':
         
            x_flat = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
            prompt_normed = self.prompt_norm(combined_prompt.unsqueeze(1))  # [B, 1, prompt_dim]
           
            q = self.cross_attn_q(x_flat) * feat_weight  # [B, H*W, C]
  
            k = self.cross_attn_k(prompt_normed) * prompt_weight  # [B, 1, C]
            v = self.cross_attn_v(prompt_normed) * prompt_weight  # [B, 1, C]
            
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)  # [B, H*W, 1]
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)  # [B, H*W, C]
            attn_out = self.cross_attn_out(attn_out)
            
            x = x * feat_weight + attn_out.transpose(1, 2).reshape(B, C, H, W)
            
        elif self.fusion_method == 'adaptive_modulation':
            modulation = self.prompt_mlp(combined_prompt)  # [B, C*2]
            scale, shift = modulation.chunk(2, dim=1)  # [B, C], [B, C]
            scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
   
            x = x * feat_weight * (1 + scale * prompt_weight) + shift * prompt_weight
            
        return x

    def forward(self, inp, combined_prompt=None):
        x = inp

        x = self.norm1(x)

        x = self.apply_prompt_fusion(x, combined_prompt)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

@ARCH_REGISTRY.register()
class DualPromptNAFNetv3(nn.Module):
    def __init__(self, 
                 img_channel=3, 
                 width=32, 
                 middle_blk_num=12, 
                 enc_blk_nums=[2, 2, 4, 8], 
                 dec_blk_nums=[2, 2, 2, 2],
                 # Prompt相关参数
                 use_prompt=True,
                 domain_pool_size=6,  
                 degradation_pool_size=12,  
                 domain_prompt_dim=1024,
                 degradation_prompt_dim=1024,
                 domain_top_k=3,
                 degradation_top_k=3,
                 value_num_tokens=5,  
                 prompt_fusion_method='cross_attention',  
                 fusion_method='cross_attention'  
                ):
        
        super().__init__()
        
        self.use_prompt = use_prompt
        self.fusion_method = fusion_method
        self.prompt_fusion_method = prompt_fusion_method 

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[PromptEnhancedNAFBlock(chan, use_prompt=False) for _ in range(num)] 
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        final_encoder_channels = chan
   
        if use_prompt:
            if prompt_fusion_method == 'concat':
                combined_prompt_dim = domain_prompt_dim + degradation_prompt_dim
            else:  # cross_attention
                combined_prompt_dim = domain_prompt_dim  
        else:
            combined_prompt_dim = 0
            
        self.middle_blks = nn.Sequential(
            *[PromptEnhancedNAFBlock(chan, use_prompt=use_prompt, prompt_dim=combined_prompt_dim, fusion_method=fusion_method) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[PromptEnhancedNAFBlock(chan, use_prompt=use_prompt, prompt_dim=combined_prompt_dim, fusion_method=fusion_method) for _ in range(num)]
                )
            )
        
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

            self.domain_mapper = DomainFeatureMapper(width, domain_prompt_dim)
            self.degradation_mapper = DegradationFeatureMapper(final_encoder_channels, degradation_prompt_dim)
   
            if prompt_fusion_method == 'cross_attention':
                self.prompt_cross_attention = nn.MultiheadAttention(
                    embed_dim=domain_prompt_dim,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                )
                self.prompt_fusion_norm = nn.LayerNorm(domain_prompt_dim)
                self.prompt_fusion_ffn = nn.Sequential(
                    nn.Linear(domain_prompt_dim, domain_prompt_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(domain_prompt_dim * 2, domain_prompt_dim)
                )

        self.padder_size = 2 ** len(self.encoders)

    def fuse_prompts(self, domain_prompt, degradation_prompt):

        if self.prompt_fusion_method == 'concat':

            combined_prompt = torch.cat([domain_prompt, degradation_prompt], dim=1)
        elif self.prompt_fusion_method == 'cross_attention':

            B = domain_prompt.shape[0]

            domain_prompt_seq = domain_prompt.unsqueeze(1)
            degradation_prompt_seq = degradation_prompt.unsqueeze(1)
    
            attended_prompt, _ = self.prompt_cross_attention(
                query=domain_prompt_seq,
                key=degradation_prompt_seq,
                value=degradation_prompt_seq
            )
          
            combined_prompt = self.prompt_fusion_norm(attended_prompt + domain_prompt_seq)
    
            combined_prompt = combined_prompt + self.prompt_fusion_ffn(combined_prompt)
       
            combined_prompt = combined_prompt.squeeze(1)
        else:
            raise ValueError(f"Unknown prompt fusion method: {self.prompt_fusion_method}")
            
        return combined_prompt

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        first_layer_features = x 
       
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        encoded_features = x
  
        combined_prompt = None
        if self.use_prompt:
         
            domain_query = self.domain_mapper(first_layer_features)  # [B, domain_prompt_dim]
            domain_prompt, domain_indices = self.domain_prompt_pool(domain_query)
   
            degradation_query = self.degradation_mapper(encoded_features)  # [B, degradation_prompt_dim]
            degradation_prompt, degradation_indices = self.degradation_prompt_pool(degradation_query)
            
            combined_prompt = self.fuse_prompts(domain_prompt, degradation_prompt)
            
        if self.use_prompt:
            for middle_blk in self.middle_blks:
                x = middle_blk(x, combined_prompt)
        else:
            x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            if self.use_prompt:
                for dec_blk in decoder:
                    x = dec_blk(x, combined_prompt)
            else:
                x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x