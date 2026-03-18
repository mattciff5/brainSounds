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
                 num_decoder_queries=50, 
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
    def __init__(self, num_parcels=200, max_voxels=564, output_dim=768, num_decoder_queries=50,
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

    Stable Audio si aspetta esattamente 128 token di lunghezza 768.
    Se num_queries != 128, un Linear riadatta la dimensione della sequenza.

    Architettura:
        Linear(768 → 768)  per ogni token  (feature projection)
        Linear(num_queries → 128)           (sequence length projection)
        LayerNorm(768)
    """

    def __init__(self, num_queries: int = 50, hidden_dim: int = 768, out_len: int = 128):
        super().__init__()
        self.out_len    = out_len
        self.num_queries = num_queries

        # Proiezione per features (applicata token per token)
        self.feat_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Proiezione per lunghezza sequenza  num_queries → out_len
        # Tratta la sequenza come "canali" di una conv 1D
        if num_queries != out_len:
            self.seq_proj = nn.Linear(num_queries, out_len, bias=False)
        else:
            self.seq_proj = nn.Identity()

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_queries, 768]
        Returns:
            [B, 128, 768]
        """
        # Feature projection token per token
        x = self.feat_proj(x)                          # [B, Q, 768]

        # Sequence length projection: trasposiamo per operare su Q
        x = x.transpose(1, 2)                         # [B, 768, Q]
        x = self.seq_proj(x)                          # [B, 768, 128]
        x = x.transpose(1, 2)                         # [B, 128, 768]

        x = self.norm(x)
        return x


class AudioDurationEmbedding(nn.Module):
    """
    Genera un embedding fisso per la durata target del audio.
    Stable Audio si aspetta [B, 1, 1536] per audio_duration_embeds.

    In training usiamo sempre la stessa durata → può essere un parametro fisso
    oppure un sinusoidal encoding della durata in secondi.
    """

    def __init__(self, embed_dim: int = 1536, max_duration_s: float = 10.0):
        super().__init__()
        self.embed_dim    = embed_dim
        self.max_duration = max_duration_s
        # Apprendibile: un vettore per ogni secondo discretizzato
        self.duration_embedding = nn.Embedding(
            num_embeddings=int(max_duration_s * 10) + 1,  # decimi di secondo
            embedding_dim=embed_dim,
        )

    def forward(self, duration_s: float, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            duration_s : durata in secondi (float)
            batch_size : B
            device     : device del batch
        Returns:
            [B, 1, 1536]
        """
        idx = torch.tensor(
            [int(duration_s * 10)], dtype=torch.long, device=device
        ).expand(batch_size)                             # [B]
        emb = self.duration_embedding(idx)               # [B, 1536]
        return emb.unsqueeze(1)                          # [B, 1, 1536]


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
        num_decoder_queries: int = 50,
        condition_dim: int = 768,
        sub_approach: str = "transformer_decoder",
        target_duration_s: float = 4.0,
    ):
        super().__init__()

        # ── Componenti addestrabili ─────────────────────────────────────────
        self.guidance_generator = GuidanceGenerator(
            num_parcels=num_rois,
            max_voxels=max_voxels,
            output_dim=condition_dim,
            num_decoder_queries=num_decoder_queries,
            sub_approach=sub_approach,
        )

        self.audio_proj = AudioProjModel(
            num_queries=num_decoder_queries,
            hidden_dim=condition_dim,
            out_len=128,       # Stable Audio si aspetta 128 token
        )

        self.duration_embedding = AudioDurationEmbedding(
            embed_dim=1536,
            max_duration_s=max(target_duration_s + 1, 10.0),
        )

        # ── Backbone frozen ────────────────────────────────────────────────
        self.pipe = pipe
        self._freeze_backbone()

        self.target_duration_s = target_duration_s

    def _freeze_backbone(self):
        for p in self.pipe.transformer.parameters():
            p.requires_grad_(False)
        for p in self.pipe.vae.parameters():
            p.requires_grad_(False)

    def get_trainable_params(self):
        return (
            list(self.guidance_generator.parameters()) +
            list(self.audio_proj.parameters()) +
            list(self.duration_embedding.parameters())
        )

    def encode_brain(self, brain_data: torch.Tensor) -> torch.Tensor:
        """
        brain_data: [B, num_rois, max_voxels]
        Returns:    [B, 128, 768]  pronti per il conditioner di Stable Audio
        """
        condition_tokens, _ = self.guidance_generator(brain_data)   # [B, Q, 768]
        audio_tokens        = self.audio_proj(condition_tokens)      # [B, 128, 768]
        return audio_tokens

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

        # 1) Token neurali
        prompt_embeds = self.encode_brain(brain_data)          # [B, 128, 768]

        # 2) Durata con pipe.encode_duration — stesso metodo usato in inference
        seconds_start_hs, seconds_end_hs = self.pipe.encode_duration(
            audio_start_in_s            = 0.0,
            audio_end_in_s              = self.target_duration_s,
            device                      = device,
            do_classifier_free_guidance = False,
            batch_size                  = B,
        )

        # 3) Concatena esattamente come la pipeline originale
        encoder_hidden_states = torch.cat(
            [prompt_embeds, seconds_start_hs, seconds_end_hs], dim=1
        )                                                      # [B, 130, 768]

        global_hidden_states = torch.cat(
            [seconds_start_hs, seconds_end_hs], dim=2
        )                                                      # [B, 1, 1536]

        # 4) Rotary embedding
        rotary_embedding = get_1d_rotary_pos_embed(
            self.pipe.rotary_embed_dim,
            noisy_latents.shape[2] + global_hidden_states.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        rotary_embedding = tuple(r.to(device) for r in rotary_embedding)

        # 5) Forward DiT
        noise_pred = self.pipe.transformer(
            hidden_states         = noisy_latents,
            timestep              = timesteps,
            encoder_hidden_states = encoder_hidden_states,
            global_hidden_states  = global_hidden_states,
            rotary_embedding      = rotary_embedding,
            return_dict           = False,
        )[0]
        
        return noise_pred