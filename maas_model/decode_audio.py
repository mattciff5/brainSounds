"""
decode_audio.py
===============
Genera audio a partire da dati neurali usando un checkpoint addestrato.

Esempio di utilizzo:
    python decode_audio.py \
        --data_path    /path/pooled_data.npy \
        --wav_dir      /path/wav \
        --sound_names  /path/SoundNames.npy \
        --ckpt_path    checkpoints/epoch_100.pt \
        --output_dir   outputs/ \
        --split        test
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from diffusers import StableAudioPipeline
from tqdm import tqdm
import pickle

from dataset import build_datasets, ROI_LIST
from model  import AudioNeuroAdapter
from diffusers.pipelines.stable_audio.pipeline_stable_audio import get_1d_rotary_pos_embed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",    type=str, required=True)
    p.add_argument("--wav_dir",      type=str, required=True)
    p.add_argument("--sound_names",  type=str, required=True)
    p.add_argument("--ckpt_path",    type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="./decoded_audio")
    p.add_argument("--split",        type=str, default="test",
                   choices=["train", "test"])
    p.add_argument("--cv",           type=str, default="CV2")
    p.add_argument("--stable_audio_id", type=str,
                   default="stabilityai/stable-audio-open-1.0")
    p.add_argument("--num_decoder_queries", type=int, default=128)
    p.add_argument("--target_duration_s",   type=float, default=4.0)
    p.add_argument("--target_sr",           type=int,   default=44100)
    p.add_argument("--conditioning_mode",   type=str, default="brain_only",
                   choices=["brain_only", "empty_prompt_plus_brain", "empty_prompt_ip_adapter"])
    p.add_argument("--num_inference_steps", type=int,   default=100)
    p.add_argument("--batch_size",          type=int,   default=1)
    p.add_argument("--average_test_repeats", action="store_true",
                   help="Media i campioni test ripetuti per suono prima della decodifica.")
    p.add_argument("--save_groundtruth",    action="store_true",
                   help="Salva anche i file audio originali per confronto")
    return p.parse_args()


@torch.no_grad()
def decode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
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

    # ── Carica Stable Audio ────────────────────────────────────────────────
    print("Caricamento Stable Audio...")
    pipe = StableAudioPipeline.from_pretrained(
        stable_audio_id, torch_dtype=torch.float32
    ).to(device)
    vae       = pipe.vae.to(device).eval()
    scheduler = pipe.scheduler

    # ── Carica modello con checkpoint ─────────────────────────────────────
    model = AudioNeuroAdapter(
        pipe                = pipe,
        num_rois            = len(ROI_LIST),
        max_voxels          = 1024,
        num_decoder_queries = num_decoder_queries,
        target_duration_s   = target_duration_s,
        conditioning_mode   = conditioning_mode,
        train_backbone_cross_attention = train_backbone_cross_attention,
        train_cross_attention_proj = train_cross_attention_proj,
    ).to(device)

    model.guidance_generator.load_state_dict(ckpt["guidance_generator"])
    model.audio_proj.load_state_dict(ckpt["audio_proj"])
    if model.ip_adapter_modules is not None and ckpt.get("ip_adapter_modules") is not None:
        model.ip_adapter_modules.load_state_dict(ckpt["ip_adapter_modules"])
    if train_backbone_cross_attention and ckpt.get("trainable_backbone") is not None:
        model.pipe.transformer.load_state_dict(ckpt["trainable_backbone"])
    model.eval()
    print(f"Checkpoint caricato: {args.ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    # ── Dataset ────────────────────────────────────────────────────────────
    with open(args.data_path, "rb") as f:
        pooled_data = pickle.load(f)
    sound_names = np.load(args.sound_names, allow_pickle=True)

    train_ds, test_ds = build_datasets(
        pooled_data  = pooled_data,
        cv           = cv,
        wav_dir      = args.wav_dir,
        sound_names  = sound_names,
        target_sr    = target_sr,
        target_len_s = target_duration_s,
        average_test_repeats = args.average_test_repeats and args.split == "test",
    )
    dataset = test_ds if args.split == "test" else train_ds
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Inference loop ─────────────────────────────────────────────────────
    scheduler.set_timesteps(args.num_inference_steps)

    sample_idx = 0
    for batch in tqdm(loader, desc="Decoding"):
        brain     = batch["brain_data"].to(device)     # [B, 6, 1024]
        sound_idx = batch["sound_idx"]                 # [B]

        B = brain.shape[0]

        text_audio_duration_embeds, audio_duration_embeds = model.build_conditioning(
            brain_data=brain,
            device=device,
        )

        # 2) Latenti di partenza
        waveform_length = int(pipe.transformer.config.sample_size)
        latent_shape    = (B, pipe.transformer.config.in_channels, waveform_length)
        latents = torch.randn(latent_shape, device=device, dtype=torch.float32)
        latents = latents * scheduler.init_noise_sigma

        # 3) Rotary embedding — CRITICO
        rotary_embedding = get_1d_rotary_pos_embed(
            pipe.rotary_embed_dim,
            latents.shape[2] + audio_duration_embeds.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        # porta su device
        rotary_embedding = tuple(r.to(device) for r in rotary_embedding)

        # 4) Denoising loop
        scheduler.set_timesteps(args.num_inference_steps, device=device)

        for i, t in enumerate(scheduler.timesteps):
            latent_input = scheduler.scale_model_input(latents, t)

            noise_pred = model.pipe.transformer(
                hidden_states         = latent_input,
                timestep              = t.unsqueeze(0).expand(B),
                encoder_hidden_states = text_audio_duration_embeds,
                global_hidden_states  = audio_duration_embeds,
                rotary_embedding      = rotary_embedding,
                return_dict           = False,
            )[0]

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 5) Decode → waveform
        waveform_start = 0
        waveform_end   = int(target_duration_s * pipe.vae.config.sampling_rate)

        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
        audio_out = vae.decode(latents / scaling_factor).sample       # [B, 2, samples_full]
        audio_out = audio_out[:, :, waveform_start:waveform_end]      # tronca alla durata esatta
        audio_out = audio_out.mean(dim=1, keepdim=True).cpu()         # [B, 1, samples] mono

        # 5) Salva
        for i in range(B):
            wav    = audio_out[i]          # [1, 44100]
            s_idx  = int(sound_idx[i])
            out_path = os.path.join(args.output_dir, f"pred_sound{s_idx:04d}_sample{sample_idx:04d}.wav")
            torchaudio.save(out_path, wav, target_sr)

            if args.save_groundtruth:
                gt_wav  = batch["audio_target"][i].unsqueeze(0)   # [1, 44100]
                gt_path = os.path.join(args.output_dir, f"gt_sound{s_idx:04d}_sample{sample_idx:04d}.wav")
                torchaudio.save(gt_path, gt_wav, target_sr)

            sample_idx += 1

    print(f"\nDecodifica completata. {sample_idx} file salvati in: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    decode(args)
