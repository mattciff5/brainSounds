"""
train_audio_neuro_adapter.py
============================
Training loop per AudioNeuroAdapter.

Flusso:
  1. Carica Stable Audio (frozen)
  2. Carica i dati neurali + audio
  3. Per ogni batch:
       a. Encode audio reale → latenti con VAE
       b. Aggiungi rumore DDPM ai latenti
       c. Predici il rumore con AudioNeuroAdapter condizionato sui dati neurali
       d. MSE loss su predizione di rumore (+ opzionale CLAP perceptual loss)
  4. Ottimizza solo GuidanceGenerator + AudioProjModel
"""

import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# diffusers
from diffusers import StableAudioPipeline

# locale
from dataset import build_datasets, ROI_LIST
from model import AudioNeuroAdapter


# ─── Argomenti ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",      type=str, required=True,
                   help="Path al file .npy con pooled_data")
    p.add_argument("--wav_dir",        type=str, required=True,
                   help="Cartella con i file .wav")
    p.add_argument("--sound_names",    type=str, required=True,
                   help="Path a SoundNames.npy")
    p.add_argument("--cv",             type=str, default="CV2")
    p.add_argument("--output_dir",     type=str, default="./checkpoints")

    p.add_argument("--stable_audio_id", type=str,
                   default="stabilityai/stable-audio-open-1.0")
    p.add_argument("--num_decoder_queries", type=int, default=16)
    p.add_argument("--target_duration_s",   type=float, default=1.0)
    p.add_argument("--target_sr",           type=int,   default=44100)
    p.add_argument("--conditioning_mode",   type=str, default="brain_only",
                   choices=["brain_only", "empty_prompt_plus_brain", "empty_prompt_ip_adapter"])
    p.add_argument("--train_backbone_cross_attention", action="store_true",
                   help="Sblocca solo la cross-attention del DiT di Stable Audio (attn2 + norm2).")
    p.add_argument("--train_cross_attention_proj", action="store_true",
                   help="Sblocca anche cross_attention_proj del DiT insieme alla cross-attention.")

    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--num_epochs",     type=int,   default=50)
    p.add_argument("--grad_accum",     type=int,   default=1)
    p.add_argument("--warmup_steps",   type=int,   default=100)
    p.add_argument("--save_every",     type=int,   default=10)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--min_snr_gamma",  type=float, default=5.0,
                   help="Gamma per Min-SNR weighting. Usa <=0 per disabilitarlo.")
    p.add_argument("--cfg_dropout_prob", type=float, default=0.05,
                   help="Probabilità di usare empty prompt al posto del brain conditioning durante il training (CFG dropout). Default 0.05.")
    return p.parse_args()


# ─── Utility ──────────────────────────────────────────────────────────────────

def encode_audio_to_latents(vae, audio: torch.Tensor, device) -> torch.Tensor:
    """
    Encode waveform → latenti usando il VAE di Stable Audio.

    Args:
        audio: [B, samples]  float32 in [-1, 1]
    Returns:
        latents: [B, C_lat, T_lat]
    """
    # Stable Audio VAE si aspetta [B, 1, samples] (mono)
    audio_in = audio.unsqueeze(1).to(device)          # [B, 1, samples]
    audio_in = audio_in.expand(-1, 2, -1)
    with torch.no_grad():
        latents = vae.encode(audio_in).latent_dist.sample()
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
        latents = latents * scaling_factor
    return latents   # [B, C_lat, T_lat]


def get_scheduler_sigmas(noise_scheduler, timesteps, n_dim, dtype, device):
    if not hasattr(noise_scheduler, "sigmas") or not hasattr(noise_scheduler, "timesteps"):
        raise AttributeError("Lo scheduler non espone sigmas/timesteps, impossibile usare il percorso EDM/Cosine.")

    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)

    step_indices = [noise_scheduler.index_for_timestep(t, schedule_timesteps) for t in timesteps]
    sigma = sigmas[step_indices].flatten()

    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)

    return sigma


def sample_training_timesteps(noise_scheduler, batch_size, device):
    # noise_scheduler.timesteps must already be set to 1000 steps before the
    # training loop starts (done once in train()).  We just sample from them.
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    indices = torch.randint(0, len(schedule_timesteps), (batch_size,), device=device)
    return schedule_timesteps[indices]


def scale_model_input_for_training(noise_scheduler, noisy_latents, timesteps):
    if hasattr(noise_scheduler, "precondition_inputs") and hasattr(noise_scheduler, "sigmas"):
        sigma = get_scheduler_sigmas(
            noise_scheduler,
            timesteps=timesteps,
            n_dim=noisy_latents.ndim,
            dtype=noisy_latents.dtype,
            device=noisy_latents.device,
        )
        return noise_scheduler.precondition_inputs(noisy_latents, sigma)

    if timesteps.ndim > 0 and timesteps.numel() > 1:
        unique_timesteps = torch.unique(timesteps)
        if unique_timesteps.numel() != 1:
            raise ValueError(
                "Questo scheduler richiede un solo timestep per batch durante scale_model_input()."
            )
        timestep = unique_timesteps[0]
    else:
        timestep = timesteps

    return noise_scheduler.scale_model_input(noisy_latents, timestep)


def get_diffusion_target(noise_scheduler, latents, noise, timesteps):
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        if not hasattr(noise_scheduler, "get_velocity"):
            sigma_data = getattr(noise_scheduler.config, "sigma_data", 1.0)
            sigma = get_scheduler_sigmas(
                noise_scheduler,
                timesteps=timesteps,
                n_dim=latents.ndim,
                dtype=latents.dtype,
                device=latents.device,
            )
            return (sigma_data * noise - (sigma / sigma_data) * latents) / torch.sqrt(sigma**2 + sigma_data**2)
        return noise_scheduler.get_velocity(latents, noise, timesteps)
    if prediction_type == "sample":
        return latents
    raise ValueError(f"prediction_type non supportato: {prediction_type}")


def get_loss_weights(noise_scheduler, timesteps, latents, gamma: float):
    if gamma is None or gamma <= 0:
        return torch.ones_like(timesteps, dtype=latents.dtype, device=latents.device)

    if hasattr(noise_scheduler, "alphas_cumprod"):
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
        snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps] + 1e-8)
    else:
        sigma_data = getattr(noise_scheduler.config, "sigma_data", 1.0)
        sigma = get_scheduler_sigmas(
            noise_scheduler,
            timesteps=timesteps,
            n_dim=1,
            dtype=latents.dtype,
            device=latents.device,
        ).squeeze(-1)
        snr = (sigma_data / sigma) ** 2

    gamma_t = torch.full_like(snr, gamma)
    return torch.minimum(snr, gamma_t) / (snr + 1e-8)


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Carica Stable Audio ─────────────────────────────────────────────
    print("Caricamento Stable Audio...")
    pipe = StableAudioPipeline.from_pretrained(
        args.stable_audio_id,
        torch_dtype=torch.float32,
    ).to(device)

    vae = pipe.vae.to(device).eval()
    noise_scheduler = pipe.scheduler

    # ── 2. Costruisci il modello ───────────────────────────────────────────
    model = AudioNeuroAdapter(
        pipe                  = pipe,        # interno viene frozen
        num_rois              = len(ROI_LIST),
        max_voxels            = 1024,
        num_decoder_queries   = args.num_decoder_queries,
        target_duration_s     = args.target_duration_s,
        conditioning_mode     = args.conditioning_mode,
        train_backbone_cross_attention = args.train_backbone_cross_attention,
        train_cross_attention_proj = args.train_cross_attention_proj,
    ).to(device)

    trainable_params = model.get_trainable_params()
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Parametri addestrabili: {n_params:,}")

    # ── 3. Dataset e DataLoader ────────────────────────────────────────────
    with open(args.data_path, "rb") as f:
        pooled_data = pickle.load(f)
    sound_names  = np.load(args.sound_names, allow_pickle=True)

    train_ds, test_ds = build_datasets(
        pooled_data   = pooled_data,
        cv            = args.cv,
        wav_dir       = args.wav_dir,
        sound_names   = sound_names,
        target_sr     = args.target_sr,
        target_len_s  = args.target_duration_s,
    )

    # Store brain normalizer stats inside the model so they are saved in the checkpoint.
    model.set_brain_normalizer(train_ds.brain_normalizer)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
    )
    # ── 4. Ottimizzatore e scheduler ──────────────────────────────────────
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.num_epochs // args.grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── 5. Training loop ───────────────────────────────────────────────────
    # Set the scheduler to 1000 steps once: this gives the correct non-uniform
    # timestep distribution for EDM/Cosine schedulers during training.
    noise_scheduler.set_timesteps(1000, device=device)
    global_step = 0

    for epoch in range(args.num_epochs):
        model.guidance_generator.train()
        model.audio_proj.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(pbar):
            brain  = batch["brain_data"].to(device)    # [B, 6, 1024]
            audio  = batch["audio_target"].to(device)  # [B, samples]
            B = brain.shape[0]

            # CFG conditioning dropout: randomly replace brain signal with empty prompt
            # so the model learns to denoise unconditionally too (required for CFG at decode).
            if args.cfg_dropout_prob > 0.0:
                drop_mask = torch.rand(B) < args.cfg_dropout_prob
                if drop_mask.any():
                    brain = brain.clone()
                    brain[drop_mask] = 0.0  # zeroed brain → guidance_generator maps to near-zero tokens

            # a) Encode audio → latenti
            latents = encode_audio_to_latents(vae, audio, device)   # [B, C, T]

            timesteps = sample_training_timesteps(
                noise_scheduler=noise_scheduler,
                batch_size=B,
                device=device,
            )

            noise = torch.randn_like(latents)
            noisy_lat = noise_scheduler.add_noise(latents, noise, timesteps)
            model_input = scale_model_input_for_training(
                noise_scheduler=noise_scheduler,
                noisy_latents=noisy_lat,
                timesteps=timesteps,
            )

            noise_pred = model(brain, model_input, timesteps)
            target = get_diffusion_target(noise_scheduler, latents, noise, timesteps)
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=tuple(range(1, loss.ndim)))
            loss = loss * get_loss_weights(noise_scheduler, timesteps, latents, args.min_snr_gamma)
            loss = loss.mean()
            loss = loss / args.grad_accum
            loss.backward()

            total_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

        # ── Salva checkpoint ────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "epoch":             epoch + 1,
                "model_config": {
                    "num_rois": len(ROI_LIST),
                    "max_voxels": 1024,
                    "num_decoder_queries": args.num_decoder_queries,
                    "target_duration_s": args.target_duration_s,
                    "target_sr": args.target_sr,
                    "stable_audio_id": args.stable_audio_id,
                    "cv": args.cv,
                    "conditioning_mode": args.conditioning_mode,
                    "min_snr_gamma": args.min_snr_gamma,
                    "cfg_dropout_prob": args.cfg_dropout_prob,
                    "train_backbone_cross_attention": args.train_backbone_cross_attention,
                    "train_cross_attention_proj": args.train_cross_attention_proj,
                },
                "guidance_generator": model.guidance_generator.state_dict(),
                "audio_proj":        model.audio_proj.state_dict(),
                "ip_adapter_scale":  model.ip_adapter_scale.detach().cpu(),
                "brain_prompt_scale": model.brain_prompt_scale.detach().cpu(),
                "ip_adapter_modules": model.ip_adapter_modules.state_dict() if model.ip_adapter_modules is not None else None,
                "trainable_backbone": model.pipe.transformer.state_dict() if args.train_backbone_cross_attention else None,
                "optimizer":         optimizer.state_dict(),
                "loss":              avg_loss,
            }, ckpt_path)
            print(f"  Checkpoint salvato: {ckpt_path}")

    print("Training completato.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
