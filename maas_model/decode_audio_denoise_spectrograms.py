"""
decode_audio_denoise_spectrograms.py
====================================

Decodifica audio da dati neurali e salva, per ciascun trial selezionato,
una figura composta con gli spettrogrammi lungo il processo di denoising.

Esempio:
    python decode_audio_denoise_spectrograms.py \
        --data_path /path/pooled_data.pkl \
        --wav_dir /path/wav \
        --sound_names /path/SoundNames.npy \
        --ckpt_path checkpoints/epoch_100.pt \
        --figure_dir outputs/denoise_figures \
        --split test \
        --samples 0-5 \
        --num_inference_steps 100 \
        --snapshot_every 10
"""

import argparse
import copy
import math
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from matplotlib.animation import FuncAnimation, PillowWriter
from diffusers import StableAudioPipeline
from diffusers.pipelines.stable_audio.pipeline_stable_audio import get_1d_rotary_pos_embed
from diffusers.models.embeddings import apply_rotary_emb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import ROI_LIST, build_datasets
from model import AudioNeuroAdapter, StableAudioIPAttnProcessor2_0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--wav_dir", type=str, required=True)
    p.add_argument("--sound_names", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--figure_dir", type=str, required=True,
                   help="Cartella dove salvare le figure composte, una per trial.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Cartella opzionale per salvare anche il wav finale decodificato.")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--cv", type=str, default="CV2")
    p.add_argument("--stable_audio_id", type=str, default="stabilityai/stable-audio-open-1.0")
    p.add_argument("--num_decoder_queries", type=int, default=16)
    p.add_argument("--target_duration_s", type=float, default=1.0)
    p.add_argument("--target_sr", type=int, default=44100)
    p.add_argument("--conditioning_mode", type=str, default="brain_only",
                   choices=["brain_only", "empty_prompt_plus_brain", "empty_prompt_ip_adapter"])
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--average_test_repeats", action="store_true")
    p.add_argument("--samples", type=str, default="0-5",
                   help="Trial 0-based da processare. Esempi: 0-5 | 0,3,8,11,14,20")
    p.add_argument("--snapshot_every", type=int, default=10,
                   help="Salva uno snapshot ogni N step di denoising.")
    p.add_argument("--seed_offset", type=int, default=55,
                   help="Offset del seed usato per inizializzare i latenti.")
    p.add_argument("--max_cols", type=int, default=4,
                   help="Numero massimo di colonne nella figura composta.")
    p.add_argument("--include_groundtruth", action="store_true",
                   help="Aggiunge un pannello con lo spettrogramma del target audio.")
    p.add_argument("--save_grid_figure", action="store_true",
                   help="Salva anche la figura composta statica con tutti gli snapshot.")
    p.add_argument("--save_gif", action="store_true",
                   help="Salva una GIF per trial: predizione a sinistra, ground truth fisso a destra.")
    p.add_argument("--gif_dir", type=str, default=None,
                   help="Cartella dove salvare le GIF. Se assente, usa figure_dir/gifs.")
    p.add_argument("--gif_fps", type=int, default=2,
                   help="Frame per secondo della GIF.")
    p.add_argument("--include_attention", action="store_true",
                   help="Aggiunge alla GIF un pannello con attenzione ROI: mapper fisso e backbone variabile.")
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--save_final_wav", action="store_true",
                   help="Salva anche il wav finale predetto nella output_dir.")
    return p.parse_args()


def parse_sample_selection(sample_str: str, dataset_len: int) -> list[int]:
    if sample_str.lower() == "all":
        return list(range(dataset_len))

    selected: list[int] = []
    for chunk in sample_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start_i = int(start_s)
            end_i = int(end_s)
            selected.extend(range(start_i, end_i + 1))
        else:
            selected.append(int(chunk))

    selected = sorted(set(selected))
    for idx in selected:
        if idx < 0 or idx >= dataset_len:
            raise ValueError(f"sample_idx fuori range: {idx} (dataset_len={dataset_len})")
    return selected


class MapperAttentionRecorder:
    def __init__(self, token_mapper):
        self.token_mapper = token_mapper
        self.handles = []
        self.records = []

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self.records.append(output[1].detach().cpu())

    def __enter__(self):
        for layer in self.token_mapper.transformer.decoder.layers:
            self.handles.append(layer.multihead_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def consume(self):
        if not self.records:
            return None
        stacked = torch.stack(self.records, dim=0)
        self.records = []
        return stacked


class AttentionStore:
    def __init__(self):
        self.cross = defaultdict(list)
        self.ip = defaultdict(list)

    def clear(self):
        self.cross.clear()
        self.ip.clear()


def _repeat_kv_to_all_heads(tensor: torch.Tensor, attn_heads: int) -> torch.Tensor:
    kv_heads = tensor.shape[1]
    if kv_heads == attn_heads:
        return tensor
    heads_per_kv_head = attn_heads // kv_heads
    return torch.repeat_interleave(
        tensor,
        heads_per_kv_head,
        dim=1,
        output_size=kv_heads * heads_per_kv_head,
    )


def _attention_probs(query, key, attn_mask=None):
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    return torch.softmax(scores.float(), dim=-1).to(query.dtype)


class RecordingStableAudioAttnProcessor2_0(nn.Module):
    def __init__(self, store: AttentionStore, name: str):
        super().__init__()
        self.store = store
        self.name = name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
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

        key = _repeat_kv_to_all_heads(key, attn.heads)
        value = _repeat_kv_to_all_heads(value, attn.heads)

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

        probs = _attention_probs(query, key, attention_mask)
        if attn.is_cross_attention:
            self.store.cross[self.name].append(probs.detach().cpu())

        hidden_states = torch.matmul(probs, value)
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


class RecordingStableAudioIPAttnProcessor2_0(nn.Module):
    def __init__(self, base_processor: StableAudioIPAttnProcessor2_0, store: AttentionStore, name: str):
        super().__init__()
        self.hidden_size = base_processor.hidden_size
        self.cross_attention_dim = base_processor.cross_attention_dim
        self.num_tokens = base_processor.num_tokens
        self.scale = base_processor.scale
        self.to_k_ip = copy.deepcopy(base_processor.to_k_ip)
        self.to_v_ip = copy.deepcopy(base_processor.to_v_ip)
        self.store = store
        self.name = name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
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
            full_encoder_hidden_states = encoder_hidden_states
            encoder_hidden_states = full_encoder_hidden_states[:, :split_idx, :]
            ip_hidden_states = full_encoder_hidden_states[:, split_idx:, :]
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

        key = _repeat_kv_to_all_heads(key, attn.heads)
        value = _repeat_kv_to_all_heads(value, attn.heads)

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

        probs = _attention_probs(query, key, attention_mask)
        hidden_states = torch.matmul(probs, value)

        if ip_hidden_states is not None and ip_hidden_states.shape[1] > 0:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            ip_key = _repeat_kv_to_all_heads(ip_key, attn.heads)
            ip_value = _repeat_kv_to_all_heads(ip_value, attn.heads)

            if attn.norm_k is not None:
                ip_key = attn.norm_k(ip_key)

            ip_probs = _attention_probs(query, ip_key, None)
            self.store.ip[self.name].append(ip_probs.detach().cpu())
            ip_hidden_states = torch.matmul(ip_probs, ip_value)
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


def swap_in_recording_processors(model: AudioNeuroAdapter, store: AttentionStore):
    transformer = model.pipe.transformer
    original = copy.deepcopy(transformer.attn_processors)
    wrapped = {}
    for name, proc in transformer.attn_processors.items():
        if ".attn1." in name:
            wrapped[name] = proc
        elif isinstance(proc, StableAudioIPAttnProcessor2_0):
            wrapped[name] = RecordingStableAudioIPAttnProcessor2_0(proc, store, name)
        else:
            wrapped[name] = RecordingStableAudioAttnProcessor2_0(store, name)
    transformer.set_attn_processor(wrapped)
    return original


def restore_processors(model: AudioNeuroAdapter, original_processors):
    model.pipe.transformer.set_attn_processor(original_processors)


def interpolation_matrix(in_len: int, out_len: int, device: torch.device) -> torch.Tensor:
    if in_len == out_len:
        return torch.eye(in_len, device=device)
    basis = torch.eye(in_len, device=device).unsqueeze(0)
    resized = F.interpolate(
        basis.transpose(1, 2),
        size=out_len,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    return resized.squeeze(0)


def get_brain_token_start_end(model: AudioNeuroAdapter):
    n_brain = model.audio_proj.out_len
    if model.conditioning_mode in {"brain_only", "empty_prompt_plus_brain"}:
        return 0, n_brain
    if model.conditioning_mode == "empty_prompt_ip_adapter":
        return None, None
    raise ValueError(f"conditioning_mode non supportata: {model.conditioning_mode}")


def compute_mapper_attention_profiles(model, brain, device):
    n_queries = model.guidance_generator.token_mapper.decoder_queries.num_embeddings
    n_brain_tokens = model.audio_proj.out_len
    query_to_projected = interpolation_matrix(n_queries, n_brain_tokens, device).detach().cpu()

    with MapperAttentionRecorder(model.guidance_generator.token_mapper) as mapper_rec:
        _condition_tokens, _ = model.guidance_generator(brain)
    mapper_attn = mapper_rec.consume()
    if mapper_attn is None:
        raise RuntimeError("Non sono riuscito a catturare l'attenzione del TokenMapper.")

    mapper_attn = mapper_attn.mean(dim=0)
    if mapper_attn.ndim == 4:
        mapper_attn = mapper_attn.mean(dim=1)

    mapper_projected = torch.einsum("tq,bqr->btr", query_to_projected, mapper_attn.cpu())
    mapper_roi = mapper_attn[0].mean(dim=0).numpy()
    return mapper_projected, mapper_roi


def compute_backbone_roi_profile(model, attn_store, mapper_projected):
    bank = attn_store.ip if model.conditioning_mode == "empty_prompt_ip_adapter" else attn_store.cross
    if not bank:
        return None

    layer_tensors = []
    for _, tensors in sorted(bank.items()):
        if not tensors:
            continue
        layer_probs = torch.stack(tensors, dim=0).mean(dim=0)
        if model.conditioning_mode != "empty_prompt_ip_adapter":
            start, end = get_brain_token_start_end(model)
            layer_probs = layer_probs[..., start:end]
        layer_tensors.append(layer_probs)

    if not layer_tensors:
        return None

    backbone_attn = torch.stack(layer_tensors, dim=0).mean(dim=0)
    backbone_attn = backbone_attn.mean(dim=1)
    transported = torch.einsum("blt,btr->blr", backbone_attn.cpu(), mapper_projected)
    return transported[0].mean(dim=0).numpy()


def load_model_and_pipe(args, device):
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model_config = ckpt.get("model_config", {})

    stable_audio_id = model_config.get("stable_audio_id", args.stable_audio_id)
    num_decoder_queries = model_config.get("num_decoder_queries", args.num_decoder_queries)
    target_duration_s = model_config.get("target_duration_s", args.target_duration_s)
    target_sr = model_config.get("target_sr", args.target_sr)
    cv = model_config.get("cv", args.cv)
    conditioning_mode = model_config.get("conditioning_mode", args.conditioning_mode)
    train_backbone_cross_attention = model_config.get("train_backbone_cross_attention", False)
    train_cross_attention_proj = model_config.get("train_cross_attention_proj", False)

    print("Caricamento Stable Audio...")
    pipe = StableAudioPipeline.from_pretrained(
        stable_audio_id,
        torch_dtype=torch.float32,
    ).to(device)

    model = AudioNeuroAdapter(
        pipe=pipe,
        num_rois=6,
        max_voxels=1024,
        num_decoder_queries=num_decoder_queries,
        target_duration_s=target_duration_s,
        conditioning_mode=conditioning_mode,
        train_backbone_cross_attention=train_backbone_cross_attention,
        train_cross_attention_proj=train_cross_attention_proj,
    ).to(device)

    model.guidance_generator.load_state_dict(ckpt["guidance_generator"])
    model.audio_proj.load_state_dict(ckpt["audio_proj"])
    if ckpt.get("ip_adapter_scale") is not None:
        model.ip_adapter_scale.data.copy_(
            ckpt["ip_adapter_scale"].to(device=device, dtype=model.ip_adapter_scale.dtype)
        )
    if ckpt.get("brain_prompt_scale") is not None:
        model.brain_prompt_scale.data.copy_(
            ckpt["brain_prompt_scale"].to(device=device, dtype=model.brain_prompt_scale.dtype)
        )
    if model.ip_adapter_modules is not None and ckpt.get("ip_adapter_modules") is not None:
        model.ip_adapter_modules.load_state_dict(ckpt["ip_adapter_modules"])
    if train_backbone_cross_attention and ckpt.get("trainable_backbone") is not None:
        model.pipe.transformer.load_state_dict(ckpt["trainable_backbone"])
    if ckpt.get("_brain_norm_mean") is not None:
        model._brain_norm_mean = ckpt["_brain_norm_mean"].to(device)
        model._brain_norm_std = ckpt["_brain_norm_std"].to(device)
    model.eval()

    print(f"Checkpoint caricato: {args.ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt, model_config, pipe, model, target_duration_s, target_sr, cv


def build_dataset(args, model, cv, target_sr, target_duration_s):
    with open(args.data_path, "rb") as f:
        pooled_data = pickle.load(f)
    sound_names = np.load(args.sound_names, allow_pickle=True)
    saved_normalizer = model.get_brain_normalizer()

    train_ds, test_ds = build_datasets(
        pooled_data=pooled_data,
        cv=cv,
        wav_dir=args.wav_dir,
        sound_names=sound_names,
        target_sr=target_sr,
        target_len_s=target_duration_s,
        average_test_repeats=args.average_test_repeats and args.split == "test",
        brain_normalizer_override=saved_normalizer,
    )
    dataset = test_ds if args.split == "test" else train_ds
    return dataset, sound_names


def decode_latents_to_waveform(pipe, latents, target_duration_s):
    waveform_end = int(target_duration_s * pipe.vae.config.sampling_rate)
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
    audio = pipe.vae.decode(latents / scaling_factor).sample
    audio = audio[:, :, :waveform_end]
    audio = audio.mean(dim=1, keepdim=True)
    return audio


def waveform_to_db_spectrogram(waveform, n_fft, hop_length):
    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )(waveform.cpu())
    spec_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(spec)
    return spec_db.squeeze(0)


def render_trial_figure(
    panels,
    figure_path,
    trial_idx,
    sound_idx,
    sound_label,
    max_cols,
):
    n_panels = len(panels)
    ncols = min(max_cols, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.ravel()

    vmin = min(float(panel["spec"].min()) for panel in panels)
    vmax = max(float(panel["spec"].max()) for panel in panels)
    last_im = None

    for ax, panel in zip(axes, panels):
        last_im = ax.imshow(
            panel["spec"].numpy(),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(panel["title"], fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Freq bin")

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"Trial {trial_idx} | sound_idx={sound_idx} | {sound_label}",
        fontsize=14,
        y=0.995,
    )
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:n_panels], fraction=0.02, pad=0.01, label="dB")
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def render_trial_gif(
    predicted_panels,
    gt_panel,
    mapper_roi,
    backbone_roi_panels,
    gif_path,
    trial_idx,
    sound_idx,
    sound_label,
    fps,
):
    if not predicted_panels:
        return

    if gt_panel is not None:
        vmin = min(
            min(float(panel["spec"].min()) for panel in predicted_panels),
            float(gt_panel["spec"].min()),
        )
        vmax = max(
            max(float(panel["spec"].max()) for panel in predicted_panels),
            float(gt_panel["spec"].max()),
        )
    else:
        vmin = min(float(panel["spec"].min()) for panel in predicted_panels)
        vmax = max(float(panel["spec"].max()) for panel in predicted_panels)

    show_attention = mapper_roi is not None and backbone_roi_panels is not None and len(backbone_roi_panels) == len(predicted_panels)

    ncols = 1 + int(gt_panel is not None) + int(show_attention)
    fig_width = 5.5 + 4.5 * (ncols - 1)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 4.6))
    if ncols == 1:
        axes = [axes]

    pred_ax = axes[0]
    pred_im = pred_ax.imshow(
        predicted_panels[0]["spec"].numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    pred_ax.set_title(predicted_panels[0]["title"], fontsize=11)
    pred_ax.set_xlabel("Frame")
    pred_ax.set_ylabel("Freq bin")

    if gt_panel is not None:
        gt_ax = axes[1]
        gt_ax.imshow(
            gt_panel["spec"].numpy(),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        gt_ax.set_title(gt_panel["title"], fontsize=11)
        gt_ax.set_xlabel("Frame")
        gt_ax.set_ylabel("Freq bin")

    mapper_bar_container = None
    roi_line_handles = {}
    mapper_ref_lines = []
    if show_attention:
        attn_ax = axes[-1]
        mapper_vals = np.asarray(mapper_roi)
        backbone_stack = np.stack([np.asarray(frame["roi"]) for frame in backbone_roi_panels], axis=0)
        max_attn = max(float(mapper_vals.max()), float(backbone_stack.max()), 1e-6)

        timestep_labels = []
        for frame in predicted_panels:
            step_part = frame["title"].replace("Step ", "")
            step_value = step_part.split("/")[0]
            timestep_labels.append(int(step_value))
        timestep_labels = np.asarray(timestep_labels, dtype=float)

        cmap = plt.get_cmap("tab10")
        for roi_idx, roi_name in enumerate(ROI_LIST):
            color = cmap(roi_idx % 10)
            ref = attn_ax.axhline(
                mapper_vals[roi_idx],
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.35,
            )
            mapper_ref_lines.append(ref)
            (line,) = attn_ax.plot(
                [timestep_labels[0]],
                [backbone_stack[0, roi_idx]],
                color=color,
                marker="o",
                linewidth=2.0,
                markersize=6,
                label=roi_name,
            )
            roi_line_handles[roi_name] = line

        attn_ax.set_xlim(timestep_labels.min(), timestep_labels.max())
        attn_ax.set_ylim(0, max_attn * 1.08)
        attn_ax.set_xlabel("Denoising step")
        attn_ax.set_ylabel("Attention")
        attn_ax.set_title(backbone_roi_panels[0]["title"], fontsize=11)
        attn_ax.grid(True, alpha=0.25)
        attn_ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            title="ROI",
            title_fontsize=9,
            frameon=True,
        )

    title = fig.suptitle(
        f"Trial {trial_idx} | sound_idx={sound_idx} | {sound_label}",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame_idx):
        panel = predicted_panels[frame_idx]
        pred_im.set_data(panel["spec"].numpy())
        pred_ax.set_title(panel["title"], fontsize=11)
        artists = [pred_im, title]
        if show_attention:
            roi_panel = backbone_roi_panels[frame_idx]
            x_vals = timestep_labels[: frame_idx + 1]
            for roi_idx, roi_name in enumerate(ROI_LIST):
                roi_line_handles[roi_name].set_data(
                    x_vals,
                    backbone_stack[: frame_idx + 1, roi_idx],
                )
            attn_ax.set_title(roi_panel["title"], fontsize=11)
            artists.extend(roi_line_handles.values())
        title.set_text(f"Trial {trial_idx} | sound_idx={sound_idx} | {sound_label}")
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(predicted_panels),
        interval=max(1, int(1000 / fps)),
        blit=False,
        repeat=True,
    )
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def maybe_save_final_wav(args, waveform, target_sr, trial_idx, sound_idx):
    if not args.save_final_wav:
        return
    if args.output_dir is None:
        raise ValueError("--save_final_wav richiede anche --output_dir.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_name = f"pred_sound{sound_idx:04d}_sample{trial_idx:04d}.wav"
    torchaudio.save(
        str(Path(args.output_dir) / out_name),
        waveform.cpu(),
        target_sr,
    )


@torch.no_grad()
def decode(args):
    if args.snapshot_every <= 0:
        raise ValueError("--snapshot_every deve essere > 0")

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)
    if args.save_gif:
        gif_dir = Path(args.gif_dir) if args.gif_dir is not None else Path(args.figure_dir) / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
    else:
        gif_dir = None

    _, _, pipe, model, target_duration_s, target_sr, cv = load_model_and_pipe(args, device)
    dataset, sound_names = build_dataset(args, model, cv, target_sr, target_duration_s)

    selected_indices = parse_sample_selection(args.samples, len(dataset))
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=1)

    print(f"Trial selezionati ({len(selected_indices)}): {selected_indices}")

    attn_store = AttentionStore() if args.include_attention else None
    original_processors = swap_in_recording_processors(model, attn_store) if args.include_attention else None

    try:
        for local_idx, batch in enumerate(tqdm(loader, desc="Saving denoise spectrograms")):
            original_trial_idx = selected_indices[local_idx]
            brain = batch["brain_data"].to(device)
            sound_idx = int(batch["sound_idx"][0])
            sound_label = str(sound_names[sound_idx])

            mapper_projected = None
            mapper_roi = None
            if args.include_attention:
                mapper_projected, mapper_roi = compute_mapper_attention_profiles(model, brain, device)

            text_audio_duration_embeds, audio_duration_embeds = model.build_conditioning(
                brain_data=brain,
                device=device,
            )

            uncond_embeds = None
            uncond_audio_duration_embeds = None
            if args.guidance_scale != 1.0:
                uncond_embeds, uncond_audio_duration_embeds = model.build_unconditional_conditioning(
                    batch_size=1,
                    device=device,
                    dtype=text_audio_duration_embeds.dtype,
                )

            scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
            scheduler.set_timesteps(args.num_inference_steps, device=device)

            waveform_length = int(pipe.transformer.config.sample_size)
            latent_shape = (1, pipe.transformer.config.in_channels, waveform_length)
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed_offset + original_trial_idx)
            latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float32)
            latents = latents * scheduler.init_noise_sigma

            rotary_embedding = get_1d_rotary_pos_embed(
                pipe.rotary_embed_dim,
                latents.shape[2] + audio_duration_embeds.shape[1],
                use_real=True,
                repeat_interleave_real=False,
            )
            rotary_embedding = tuple(r.to(device) for r in rotary_embedding)

            panels = []
            predicted_panels = []
            backbone_roi_panels = []
            gt_panel = None
            if args.include_groundtruth:
                gt_wave = batch["audio_target"][0].unsqueeze(0)
                gt_spec = waveform_to_db_spectrogram(gt_wave, args.n_fft, args.hop_length)
                gt_panel = {"title": "Ground truth", "spec": gt_spec}
                panels.append(gt_panel)

            total_steps = len(scheduler.timesteps)
            for step_idx, t in enumerate(scheduler.timesteps, start=1):
                latent_input = scheduler.scale_model_input(latents, t)

                if args.include_attention:
                    attn_store.clear()

                noise_pred_cond = model.pipe.transformer(
                    hidden_states=latent_input,
                    timestep=t.unsqueeze(0).expand(1),
                    encoder_hidden_states=text_audio_duration_embeds,
                    global_hidden_states=audio_duration_embeds,
                    rotary_embedding=rotary_embedding,
                    return_dict=False,
                )[0]

                current_backbone_roi = None
                if args.include_attention:
                    current_backbone_roi = compute_backbone_roi_profile(model, attn_store, mapper_projected)

                if args.guidance_scale == 1.0:
                    noise_pred = noise_pred_cond
                else:
                    noise_pred_uncond = model.pipe.transformer(
                        hidden_states=latent_input,
                        timestep=t.unsqueeze(0).expand(1),
                        encoder_hidden_states=uncond_embeds,
                        global_hidden_states=uncond_audio_duration_embeds,
                        rotary_embedding=rotary_embedding,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents).prev_sample

                should_capture = (step_idx % args.snapshot_every == 0) or (step_idx == total_steps)
                if should_capture:
                    waveform = decode_latents_to_waveform(pipe, latents, target_duration_s)
                    spec = waveform_to_db_spectrogram(
                        waveform[0].cpu(),
                        args.n_fft,
                        args.hop_length,
                    )
                    panels.append({
                        "title": f"Step {step_idx}/{total_steps}",
                        "spec": spec,
                    })
                    predicted_panels.append({
                        "title": f"Step {step_idx}/{total_steps}",
                        "spec": spec,
                    })
                    if args.include_attention and current_backbone_roi is not None:
                        backbone_roi_panels.append({
                            "title": f"ROI attention @ step {step_idx}/{total_steps}",
                            "roi": current_backbone_roi,
                        })
                    if step_idx == total_steps:
                        maybe_save_final_wav(
                            args,
                            waveform[0].cpu(),
                            target_sr,
                            original_trial_idx,
                            sound_idx,
                        )

            if args.save_grid_figure:
                figure_name = f"trial_{original_trial_idx:04d}_sound{sound_idx:04d}_denoise_steps.png"
                figure_path = Path(args.figure_dir) / figure_name
                render_trial_figure(
                    panels=panels,
                    figure_path=figure_path,
                    trial_idx=original_trial_idx,
                    sound_idx=sound_idx,
                    sound_label=sound_label,
                    max_cols=args.max_cols,
                )

            if args.save_gif:
                gif_name = f"trial_{original_trial_idx:04d}_sound{sound_idx:04d}_denoise_steps.gif"
                gif_path = gif_dir / gif_name
                render_trial_gif(
                    predicted_panels=predicted_panels,
                    gt_panel=gt_panel,
                    mapper_roi=mapper_roi,
                    backbone_roi_panels=backbone_roi_panels if args.include_attention else None,
                    gif_path=gif_path,
                    trial_idx=original_trial_idx,
                    sound_idx=sound_idx,
                    sound_label=sound_label,
                    fps=args.gif_fps,
                )
    finally:
        if args.include_attention and original_processors is not None:
            restore_processors(model, original_processors)

    print(f"\nFigure salvate in: {args.figure_dir}")
    if args.save_gif and gif_dir is not None:
        print(f"GIF salvate in: {gif_dir}")
    if args.save_final_wav and args.output_dir is not None:
        print(f"Wav finali salvati in: {args.output_dir}")


if __name__ == "__main__":
    decode(parse_args())
