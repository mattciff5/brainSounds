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
from diffusers import StableAudioPipeline, DDPMScheduler

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
    p.add_argument("--num_decoder_queries", type=int, default=50)
    p.add_argument("--target_duration_s",   type=float, default=4.0)
    p.add_argument("--target_sr",           type=int,   default=44100)

    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--num_epochs",     type=int,   default=100)
    p.add_argument("--grad_accum",     type=int,   default=1)
    p.add_argument("--warmup_steps",   type=int,   default=100)
    p.add_argument("--save_every",     type=int,   default=10)
    p.add_argument("--seed",           type=int,   default=42)
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
    return latents   # [B, C_lat, T_lat]


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    noise_scheduler.set_timesteps(1000, device=device)
    train_timesteps = noise_scheduler.timesteps

    # ── 2. Costruisci il modello ───────────────────────────────────────────
    model = AudioNeuroAdapter(
        pipe                  = pipe,        # interno viene frozen
        num_rois              = len(ROI_LIST),
        max_voxels            = 1024,
        num_decoder_queries   = args.num_decoder_queries,
        target_duration_s     = args.target_duration_s,
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

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 2,
    )

    # ── 4. Ottimizzatore e scheduler ──────────────────────────────────────
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.num_epochs // args.grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── 5. Training loop ───────────────────────────────────────────────────
    global_step = 0

    for epoch in range(args.num_epochs):
        model.guidance_generator.train()
        model.audio_proj.train()
        model.duration_embedding.train()

        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(pbar):
            brain  = batch["brain_data"].to(device)    # [B, 6, 1024]
            audio  = batch["audio_target"].to(device)  # [B, samples]

            # a) Encode audio → latenti
            latents = encode_audio_to_latents(vae, audio, device)   # [B, C, T]

            B = latents.shape[0]

            idx = torch.randint(0, len(train_timesteps), (1,), device=device)
            timestep = train_timesteps[idx].squeeze(0)
            timesteps = timestep.expand(B)

            noise = torch.randn_like(latents)

            sigma = timestep.to(latents.dtype)
            if sigma.ndim == 0:
                sigma = sigma.view(1, 1, 1)

            noisy_lat = latents + sigma * noise

            model_input = noise_scheduler.scale_model_input(noisy_lat, timestep)

            noise_pred = model(brain, model_input, timesteps)

            target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())
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
                "guidance_generator": model.guidance_generator.state_dict(),
                "audio_proj":        model.audio_proj.state_dict(),
                "duration_embedding": model.duration_embedding.state_dict(),
                "optimizer":         optimizer.state_dict(),
                "loss":              avg_loss,
            }, ckpt_path)
            print(f"  Checkpoint salvato: {ckpt_path}")

    print("Training completato.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
