"""
AudioNeuroAdapter
=================
Adatta NeuroAdapter (fMRI→immagine) al caso audio usando Stable Audio come backbone.

Pipeline:
  [B, num_rois, 1024]
       ↓ ParcelMapper       (riusato da NeuroAdapter)
  [B, num_rois, 768]
       ↓ TokenMapper        (riusato da NeuroAdapter)
  [B, num_queries, 768]
       ↓ AudioProjModel     (NUOVO: porta i token al formato Stable Audio)
  [B, 128, 768]             ← stessa shape di raw_t5_hidden_states
       ↓ Stable Audio DiT   (frozen, condizionato al posto del T5)
  latenti audio
       ↓ VAE decoder
  [B, samples]

Stable Audio conditioning accetta:
  - prompt_embeds          [B, 128, 768]   ← qui iniettiamo i token neurali
  - audio_duration_embeds  [B, 1, 1536]    ← fisso (durata target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── importa i moduli già scritti nel repo originale ──────────────────────────
# Assicurati che brain_adapter/ sia nel PYTHONPATH
from diffusers.pipelines.stable_audio.pipeline_stable_audio import get_1d_rotary_pos_embed
from diffusers.models.attention_processor import StableAudioAttnProcessor2_0
from transformer import Transformer


class ParcelMapper(nn.Module):
    def __init__(self, num_parcels, max_voxels, out_dim=768):
        """
        Linear projection of brain data to match the image embedding dimension.
        Projects [B, num_parcels, max_voxels] → [B, num_parcels, out_dim]
        """
        super().__init__()
        self.linear_weights = nn.Parameter(torch.empty(num_parcels, max_voxels, out_dim))
        self.linear_bias = nn.Parameter(torch.empty(num_parcels, out_dim))
        self.num_parcels = num_parcels
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_weights)
        nn.init.zeros_(self.linear_bias)

    def forward(self, x):
        # x: [B, P, V], linear_weights: [parPcels, V, 768], linear_bias: [P, 768]
        # print(f"Input shape: {x.shape}, Linear weights shape: {self.linear_weights.shape}, Linear bias shape: {self.linear_bias.shape}")
        return torch.einsum('bpv,pvd->bpd', x, self.linear_weights) + self.linear_bias


class TokenMapper(nn.Module):
    def __init__(self, 
                 num_parcels=200,
                 num_decoder_queries=128, 
                 d_model=768,
                 num_decoder_layers=1,
                 nhead=8,
                 dropout=0.1):
        super().__init__()
        
        # Learnable queries for the decoder
        self.decoder_queries = nn.Embedding(num_decoder_queries, d_model) 
        self.roi_embeddings = nn.Embedding(num_parcels, d_model) 

        self.transformer = Transformer(
            d_model=d_model, 
            dropout=dropout, 
            nhead=nhead, 
            dim_feedforward = 1024,
            num_encoder_layers=0, 
            num_decoder_layers=num_decoder_layers,
            normalize_before=True, 
            return_intermediate_enc=False,
            return_intermediate_dec=False,
            enc_output_layer=1,
        )

    def forward(self, fmri_tokens):
        """
        fmri_tokens: [B, num_fmri_tokens, d_model] 
        returns: [B, num_decoder_queries, d_model]
        """  
        B, num_parcels, d_model = fmri_tokens.shape
        num_queries = self.decoder_queries.num_embeddings

        # [B, num_parcels, d_model] → [B, d_model, num_parcels, 1]
        src = fmri_tokens.permute(0, 2, 1).unsqueeze(-1)   # [B, d_model, num_parcels, 1]

        # [num_parcels, d_model] → [1, num_parcels, d_model] → [B, num_parcels, d_model] → [B, d_model, num_parcels]
        pos_embed = self.roi_embeddings.weight.unsqueeze(0).repeat(B, 1, 1).permute(0, 2, 1).unsqueeze(-1)  # [B, d_model, num_parcels, 1]

        # [num_queries, d_model]
        query_embed = self.decoder_queries.weight    # [num_queries, d_model]
        mask = torch.zeros(B, num_parcels, device=fmri_tokens.device)  # [B, num_parcels]

        # Now call transformer exactly as before
        condition_tokens = self.transformer.forward(
            src=src,
            mask=mask,
            query_embed=query_embed,
            pos_embed=pos_embed,
        ).squeeze(0)  # [B, num_decoder_queries, d_model]

        return condition_tokens
    


class GuidanceGenerator(nn.Module):
    def __init__(self, num_parcels=200, max_voxels=564, output_dim=768, num_decoder_queries=128,
                 num_decoder_layers=1, nhead=8, dropout=0.1, sub_approach='transformer_decoder'):
        """
        Combines ParcelMapper and decoder-only TokenMapper.
        """
        super().__init__()
        self.sub_approach = sub_approach
        self.parcel_mapper = ParcelMapper(num_parcels=num_parcels, max_voxels=max_voxels, out_dim=output_dim)
        if self.sub_approach == 'transformer_decoder':
            self.token_mapper = TokenMapper(
                num_parcels=num_parcels,
                num_decoder_queries=num_decoder_queries,
                d_model=output_dim,
                num_decoder_layers=num_decoder_layers,
                nhead=nhead,
                dropout=dropout
            )
        
    def forward(self, fmri_data):
        """
        fmri_data: [B, num_parcels, max_voxels]
        Returns:
            condition_tokens: [B, num_decoder_queries, output_dim]
            fmri_tokens: [B, num_parcels, output_dim]
        """
        # B, num_parcels, max_voxels = fmri_data.shape
        # hidden_dim = 768
        # pad_size = hidden_dim - max_voxels
        # fmri_tokens = F.pad(fmri_data, (0, pad_size), "constant", 0)
        # return fmri_tokens, None
    
        fmri_tokens = self.parcel_mapper(fmri_data)
        # print(torch.isnan(self.parcel_mapper.linear_weights).any(), 'NaNs in weights')
        # print(torch.isnan(self.parcel_mapper.linear_bias).any(), 'NaNs in bias')
        # print(torch.isnan(fmri_tokens).any(), 'NaNs in fmri_tokens!')
        # print(torch.isnan(fmri_data).any(), 'NaNs in fmri_data!')
        # return fmri_tokens, None

        if self.sub_approach == 'transformer_decoder':
            condition_tokens = self.token_mapper(fmri_tokens)
            return condition_tokens, fmri_tokens
        else:
            return fmri_tokens, None 
# ─────────────────────────────────────────────────────────────────────────────


class AudioProjModel(nn.Module):
    """
    Proietta i token neurali [B, num_queries, 768] nel formato
    atteso dal conditioner di Stable Audio [B, 128, 768].

    Idealmente usiamo 128 query direttamente. Se la lunghezza non coincide,
    facciamo un resize lineare sulla dimensione di sequenza.
    """

    def __init__(self, num_queries: int = 128, hidden_dim: int = 768, out_len: int = 128):
        super().__init__()
        self.out_len    = out_len
        self.num_queries = num_queries

        self.feat_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_queries, 768]
        Returns:
            [B, 128, 768]
        """
        x = self.feat_proj(x)
        if x.shape[1] != self.out_len:
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2),
                size=self.out_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return self.norm(x)


class StableAudioIPAttnProcessor2_0(nn.Module):
    """
    Stable Audio cross-attention processor with a separate IP-Adapter-like branch.
    The last `num_tokens` encoder tokens are treated as brain/IP tokens and get
    their own key/value projections.
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int, num_tokens: int, scale: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            ip_hidden_states = None
        else:
            if encoder_hidden_states.shape[1] < self.num_tokens:
                raise ValueError(
                    f"Attesi almeno {self.num_tokens} token IP, ottenuti {encoder_hidden_states.shape[1]}."
                )
            split_idx = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :split_idx, :],
                encoder_hidden_states[:, split_idx:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        sequence_length = encoder_hidden_states.shape[1]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1, output_size=key.shape[1] * heads_per_kv_head)
            value = torch.repeat_interleave(
                value, heads_per_kv_head, dim=1, output_size=value.shape[1] * heads_per_kv_head
            )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if rotary_emb is not None:
            query_dtype = query.dtype
            key_dtype = key.dtype
            query = query.to(torch.float32)
            key = key.to(torch.float32)
            rot_dim = rotary_emb[0].shape[-1]
            query_to_rotate, query_unrotated = query[..., :rot_dim], query[..., rot_dim:]
            query_rotated = apply_rotary_emb(query_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)
            query = torch.cat((query_rotated, query_unrotated), dim=-1)
            if not attn.is_cross_attention:
                key_to_rotate, key_unrotated = key[..., :rot_dim], key[..., rot_dim:]
                key_rotated = apply_rotary_emb(key_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)
                key = torch.cat((key_rotated, key_unrotated), dim=-1)
            query = query.to(query_dtype)
            key = key.to(key_dtype)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

            if kv_heads != attn.heads:
                heads_per_kv_head = attn.heads // kv_heads
                ip_key = torch.repeat_interleave(
                    ip_key, heads_per_kv_head, dim=1, output_size=ip_key.shape[1] * heads_per_kv_head
                )
                ip_value = torch.repeat_interleave(
                    ip_value, heads_per_kv_head, dim=1, output_size=ip_value.shape[1] * heads_per_kv_head
                )

            if attn.norm_k is not None:
                ip_key = attn.norm_k(ip_key)

            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states + self.scale * ip_hidden_states

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AudioNeuroAdapter(nn.Module):
    """
    Wrapper completo: GuidanceGenerator + AudioProjModel + Stable Audio.

    Solo GuidanceGenerator e AudioProjModel vengono addestrati.
    Il modello Stable Audio è frozen.

    Args:
        stable_audio_model : modello Stable Audio caricato da diffusers
        num_rois           : numero di ROI (6 per i dati Maas)
        max_voxels         : voxel per ROI (1024)
        num_decoder_queries: token prodotti dal TokenMapper (es. 50)
        condition_dim      : dimensione dei token (768 per compatibilità CLIP/T5)
        sub_approach       : 'transformer_decoder' o 'linear_projection'
        target_duration_s  : durata audio target in secondi
    """

    def __init__(
        self,
        pipe,
        num_rois: int = 6,
        max_voxels: int = 1024,
        num_decoder_queries: int = 128,
        condition_dim: int = 768,
        sub_approach: str = "transformer_decoder",
        num_decoder_layers: int = 1,
        nhead: int = 8,
        dropout: float = 0.1,
        target_duration_s: float = 4.0,
        conditioning_mode: str = "brain_only",
        train_backbone_cross_attention: bool = False,
        train_cross_attention_proj: bool = False,
    ):
        super().__init__()

        # ── Componenti addestrabili ─────────────────────────────────────────
        self.guidance_generator = GuidanceGenerator(
            num_parcels=num_rois,
            max_voxels=max_voxels,
            output_dim=condition_dim,
            num_decoder_queries=num_decoder_queries,
            num_decoder_layers=num_decoder_layers,
            nhead=nhead,
            dropout=dropout,
            sub_approach=sub_approach,
        )

        self.conditioning_mode = conditioning_mode
        proj_out_len = 128 if conditioning_mode in {"brain_only", "empty_prompt_plus_brain"} else num_decoder_queries
        self.audio_proj = AudioProjModel(
            num_queries=num_decoder_queries,
            hidden_dim=condition_dim,
            out_len=proj_out_len,
        )
        self.brain_prompt_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("_empty_prompt_cache", torch.empty(0), persistent=False)
        self.ip_adapter_scale = nn.Parameter(torch.tensor(1.0))
        self.num_ip_tokens = proj_out_len

        # ── Backbone frozen ────────────────────────────────────────────────
        self.pipe = pipe
        self.train_backbone_cross_attention = train_backbone_cross_attention
        self.train_cross_attention_proj = train_cross_attention_proj
        self.ip_adapter_modules = None
        if self.conditioning_mode == "empty_prompt_ip_adapter":
            self._setup_ip_adapter_processors()
        self._freeze_backbone()

        self.target_duration_s = target_duration_s

    def _freeze_backbone(self):
        for p in self.pipe.transformer.parameters():
            p.requires_grad_(False)
        for p in self.pipe.vae.parameters():
            p.requires_grad_(False)
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            for p in self.pipe.text_encoder.parameters():
                p.requires_grad_(False)
        if hasattr(self.pipe, "projection_model") and self.pipe.projection_model is not None:
            for p in self.pipe.projection_model.parameters():
                p.requires_grad_(False)
        if self.ip_adapter_modules is not None:
            for p in self.ip_adapter_modules.parameters():
                p.requires_grad_(True)
        if self.train_backbone_cross_attention:
            self._enable_cross_attention_training()

    def _resolve_module(self, root: nn.Module, path: str) -> nn.Module:
        module = root
        for part in path.split("."):
            module = getattr(module, part)
        return module

    def _setup_ip_adapter_processors(self):
        transformer = self.pipe.transformer
        attn_procs = {}
        adapter_modules = []

        for name in transformer.attn_processors.keys():
            module_path = name.removesuffix(".processor")
            attn_module = self._resolve_module(transformer, module_path)

            if ".attn1." in name:
                attn_procs[name] = StableAudioAttnProcessor2_0()
                continue

            processor = StableAudioIPAttnProcessor2_0(
                hidden_size=attn_module.to_k.out_features,
                cross_attention_dim=attn_module.cross_attention_dim,
                num_tokens=self.num_ip_tokens,
                scale=1.0,
            )
            processor.to_k_ip.weight.data.copy_(attn_module.to_k.weight.data)
            processor.to_v_ip.weight.data.copy_(attn_module.to_v.weight.data)
            attn_procs[name] = processor
            adapter_modules.append(processor)

        transformer.set_attn_processor(attn_procs)
        self.ip_adapter_modules = nn.ModuleList(adapter_modules)

    def _enable_cross_attention_training(self):
        transformer = self.pipe.transformer

        if hasattr(transformer, "cross_attention_proj") and self.train_cross_attention_proj:
            for p in transformer.cross_attention_proj.parameters():
                p.requires_grad_(True)

        if not hasattr(transformer, "transformer_blocks"):
            raise AttributeError("Il transformer non espone transformer_blocks; impossibile sbloccare la sola cross-attention.")

        for block in transformer.transformer_blocks:
            if hasattr(block, "attn2"):
                for p in block.attn2.parameters():
                    p.requires_grad_(True)
            if hasattr(block, "norm2"):
                for p in block.norm2.parameters():
                    p.requires_grad_(True)

    def get_trainable_params(self):
        params = list(self.guidance_generator.parameters()) + list(self.audio_proj.parameters())
        if self.conditioning_mode == "empty_prompt_plus_brain":
            params.append(self.brain_prompt_scale)
        if self.conditioning_mode == "empty_prompt_ip_adapter":
            params.append(self.ip_adapter_scale)
            if self.ip_adapter_modules is not None:
                params.extend(self.ip_adapter_modules.parameters())
        if self.train_backbone_cross_attention:
            params.extend(p for p in self.pipe.transformer.parameters() if p.requires_grad)
        # Remove duplicates while preserving order.
        seen = set()
        unique_params = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)
        return unique_params

    @torch.no_grad()
    def _get_empty_prompt_embeds(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache = self._empty_prompt_cache
        if cache.numel() == 0 or cache.device != device or cache.dtype != dtype:
            empty_prompt = self.pipe.encode_prompt(
                prompt=[""],
                device=device,
                do_classifier_free_guidance=False,
            )
            if empty_prompt.shape[0] != 1:
                raise ValueError(f"Attesi embedding prompt vuoto con batch 1, ottenuto {empty_prompt.shape}.")
            self._empty_prompt_cache = empty_prompt.to(device=device, dtype=dtype)
            cache = self._empty_prompt_cache

        return cache.expand(batch_size, -1, -1)

    def encode_brain(self, brain_data: torch.Tensor) -> torch.Tensor:
        """
        brain_data: [B, num_rois, max_voxels]
        Returns:    [B, 128, 768]  pronti per il conditioner di Stable Audio
        """
        condition_tokens, _ = self.guidance_generator(brain_data)   # [B, Q, 768]
        brain_tokens        = self.audio_proj(condition_tokens)      # [B, 128, 768]

        if self.conditioning_mode in {"brain_only", "empty_prompt_plus_brain", "empty_prompt_ip_adapter"}:
            return brain_tokens

        raise ValueError(f"conditioning_mode non supportata: {self.conditioning_mode}")

    def build_conditioning(
        self,
        brain_data: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        brain_tokens = self.encode_brain(brain_data)
        batch_size = brain_data.shape[0]

        if self.conditioning_mode == "brain_only":
            prompt_embeds = brain_tokens
        elif self.conditioning_mode in {"empty_prompt_plus_brain", "empty_prompt_ip_adapter"}:
            prompt_embeds = self._get_empty_prompt_embeds(
                batch_size=batch_size,
                device=device,
                dtype=brain_tokens.dtype,
            )
            if self.conditioning_mode == "empty_prompt_plus_brain":
                prompt_embeds = prompt_embeds + self.brain_prompt_scale * brain_tokens
        else:
            raise ValueError(f"conditioning_mode non supportata: {self.conditioning_mode}")

        seconds_start_hs, seconds_end_hs = self.pipe.encode_duration(
            audio_start_in_s=0.0,
            audio_end_in_s=self.target_duration_s,
            device=device,
            do_classifier_free_guidance=False,
            batch_size=batch_size,
        )

        encoder_hidden_states = torch.cat([prompt_embeds, seconds_start_hs, seconds_end_hs], dim=1)
        if self.conditioning_mode == "empty_prompt_ip_adapter":
            encoder_hidden_states = torch.cat([encoder_hidden_states, self.ip_adapter_scale * brain_tokens], dim=1)

        global_hidden_states = torch.cat([seconds_start_hs, seconds_end_hs], dim=2)
        return encoder_hidden_states, global_hidden_states

    def forward(
        self,
        brain_data: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Un passo di denoising condizionato sui dati neurali.

        Args:
            brain_data     : [B, num_rois, 1024]
            noisy_latents  : [B, C_lat, T]   latenti audio rumorosi
            timesteps      : [B]

        Returns:
            noise_pred     : [B, C_lat, T]
        """
        B      = brain_data.shape[0]
        device = brain_data.device

        encoder_hidden_states, global_hidden_states = self.build_conditioning(
            brain_data=brain_data,
            device=device,
        )

        rotary_embedding = get_1d_rotary_pos_embed(
            self.pipe.rotary_embed_dim,
            noisy_latents.shape[2] + global_hidden_states.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        rotary_embedding = tuple(r.to(device) for r in rotary_embedding)

        noise_pred = self.pipe.transformer(
            hidden_states         = noisy_latents,
            timestep              = timesteps,
            encoder_hidden_states = encoder_hidden_states,
            global_hidden_states  = global_hidden_states,
            rotary_embedding      = rotary_embedding,
            return_dict           = False,
        )[0]
        
        return noise_pred
