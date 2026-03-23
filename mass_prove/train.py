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
import ast
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# diffusers
from diffusers import StableAudioPipeline
from diffusers.pipelines.stable_audio.pipeline_stable_audio import get_1d_rotary_pos_embed
# locale
from dataset import build_datasets, ROI_LIST
from model import AudioNeuroAdapter

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


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
    p.add_argument("--num_decoder_layers", type=int, default=1)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--target_duration_s",   type=float, default=1.0)
    p.add_argument("--target_sr",           type=int,   default=44100)
    p.add_argument("--conditioning_mode",   type=str, default="brain_only",
                   choices=["brain_only", "empty_prompt_plus_brain", "empty_prompt_ip_adapter"])
    p.add_argument("--train_backbone_cross_attention", action="store_true",
                   help="Sblocca solo la cross-attention del DiT di Stable Audio (attn2 + norm2).")
    p.add_argument("--train_cross_attention_proj", action="store_true",
                   help="Sblocca anche cross_attention_proj del DiT insieme alla cross-attention.")

    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--num_epochs",     type=int,   default=50)
    p.add_argument("--grad_accum",     type=int,   default=1)
    p.add_argument("--warmup_steps",   type=int,   default=100)
    p.add_argument("--eval_every",     type=int,   default=1,
                   help="Valuta sul test set ogni N epoche.")
    p.add_argument("--early_stop_patience", type=int, default=5,
                   help="Numero di evaluation senza miglioramento prima di fermarsi.")
    p.add_argument("--num_val_audio_samples", type=int, default=4,
                   help="Numero di sample audio di validazione da salvare quando il modello migliora.")
    p.add_argument("--val_num_inference_steps", type=int, default=50,
                   help="Numero di step di denoising per i sample audio di validazione.")
    p.add_argument("--save_val_audio_local", type=str2bool, nargs="?", const=True, default=True,
                   help="Salva i sample audio di validazione anche su disco.")
    p.add_argument("--num_workers",    type=int,   default=4,
                   help="Numero di worker per train/test DataLoader.")
    p.add_argument("--average_test_repeats", type=str2bool, nargs="?", const=True, default=False,
                   help="Media le ripetizioni del test set per suono prima della validazione.")
    p.add_argument("--wandb_project",  type=str, default="maas-audio-neuro-adapter")
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group",    type=str, default=None)
    p.add_argument("--wandb_tags",     type=str, nargs="*", default=None)
    p.add_argument("--disable_wandb",  type=str2bool, nargs="?", const=True, default=False,
                   help="Disabilita logging su Weights & Biases.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--min_snr_gamma",  type=float, default=5.0,
                   help="Gamma per Min-SNR weighting. Usa <=0 per disabilitarlo.")
    return p.parse_args()


# ─── Utility ──────────────────────────────────────────────────────────────────

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Booleano non valido: {value}")

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
    if hasattr(noise_scheduler, "timesteps"):
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        indices = torch.randint(
            low=0,
            high=len(schedule_timesteps),
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        return schedule_timesteps[indices]

    num_train_timesteps = getattr(noise_scheduler.config, "num_train_timesteps", 1000)
    return torch.randint(
        low=0,
        high=num_train_timesteps,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )


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


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def create_model_config(args):
    return {
        "num_rois": len(ROI_LIST),
        "max_voxels": 1024,
        "num_decoder_queries": args.num_decoder_queries,
        "num_decoder_layers": args.num_decoder_layers,
        "nhead": args.nhead,
        "dropout": args.dropout,
        "target_duration_s": args.target_duration_s,
        "target_sr": args.target_sr,
        "stable_audio_id": args.stable_audio_id,
        "cv": args.cv,
        "conditioning_mode": args.conditioning_mode,
        "min_snr_gamma": args.min_snr_gamma,
        "average_test_repeats": args.average_test_repeats,
        "train_backbone_cross_attention": args.train_backbone_cross_attention,
        "train_cross_attention_proj": args.train_cross_attention_proj,
    }


def maybe_init_wandb(args):
    if args.disable_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb non e' installato. Installa il pacchetto o usa --disable_wandb.")

    raw_tags = args.wandb_tags or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    if len(raw_tags) == 1 and isinstance(raw_tags[0], str):
        tag_value = raw_tags[0].strip()
        if tag_value.startswith("[") and tag_value.endswith("]"):
            try:
                parsed = ast.literal_eval(tag_value)
                if isinstance(parsed, list):
                    raw_tags = [str(tag) for tag in parsed]
            except (SyntaxError, ValueError):
                pass

    tags = list(raw_tags)
    if args.conditioning_mode not in tags:
        tags.append(args.conditioning_mode)
    if args.cv not in tags:
        tags.append(args.cv)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=tags,
        config=vars(args),
    )
    if run is not None:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("eval/*", step_metric="epoch")
        run.define_metric("samples/*", step_metric="epoch")
    return run


def maybe_log_wandb(run, payload: dict):
    if run is not None:
        run.log(payload)


@torch.no_grad()
def evaluate(model, data_loader, vae, noise_scheduler, device, min_snr_gamma: float):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        brain = batch["brain_data"].to(device)
        audio = batch["audio_target"].to(device)

        latents = encode_audio_to_latents(vae, audio, device)
        batch_size = latents.shape[0]

        timesteps = sample_training_timesteps(
            noise_scheduler=noise_scheduler,
            batch_size=batch_size,
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
        loss = loss * get_loss_weights(noise_scheduler, timesteps, latents, min_snr_gamma)
        total_loss += loss.mean().item()
        num_batches += 1

    if was_training:
        model.train()

    return total_loss / max(1, num_batches)


@torch.no_grad()
def generate_validation_samples(
    model,
    pipe,
    dataset,
    device,
    output_dir: str,
    epoch: int,
    target_sr: int,
    target_duration_s: float,
    num_samples: int,
    num_inference_steps: int,
    save_local: bool,
):
    if num_samples <= 0 or len(dataset) == 0:
        return []

    model.eval()
    sample_count = min(num_samples, len(dataset))
    sample_items = [dataset[idx] for idx in range(sample_count)]

    brain = torch.stack([item["brain_data"] for item in sample_items]).to(device)
    gt_audio = torch.stack([item["audio_target"] for item in sample_items])
    sound_idx = [int(item["sound_idx"]) for item in sample_items]

    encoder_hidden_states, global_hidden_states = model.build_conditioning(
        brain_data=brain,
        device=device,
    )

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    waveform_length = int(pipe.transformer.config.sample_size)
    latents = torch.randn(
        (sample_count, pipe.transformer.config.in_channels, waveform_length),
        device=device,
        dtype=torch.float32,
    )
    latents = latents * scheduler.init_noise_sigma

    rotary_embedding = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim,
        latents.shape[2] + global_hidden_states.shape[1],
        use_real=True,
        repeat_interleave_real=False,
    )
    rotary_embedding = tuple(r.to(device) for r in rotary_embedding)

    for t in scheduler.timesteps:
        latent_input = scheduler.scale_model_input(latents, t)
        noise_pred = pipe.transformer(
            hidden_states=latent_input,
            timestep=t.unsqueeze(0).expand(sample_count),
            encoder_hidden_states=encoder_hidden_states,
            global_hidden_states=global_hidden_states,
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
    waveform_end = int(target_duration_s * pipe.vae.config.sampling_rate)
    audio_out = pipe.vae.decode(latents / scaling_factor).sample
    audio_out = audio_out[:, :, :waveform_end].mean(dim=1, keepdim=True).cpu()

    epoch_dir = os.path.join(output_dir, "val_samples", f"epoch_{epoch:03d}") if save_local else None
    if save_local:
        os.makedirs(epoch_dir, exist_ok=True)

    records = []
    for idx in range(sample_count):
        base = f"sample{idx:02d}_sound{sound_idx[idx]:04d}"
        pred_audio = audio_out[idx].squeeze(0).numpy()
        gt_audio_np = gt_audio[idx].numpy()
        pred_path = None
        gt_path = None
        if save_local:
            pred_path = os.path.join(epoch_dir, f"{base}_pred.wav")
            gt_path = os.path.join(epoch_dir, f"{base}_gt.wav")
            torchaudio.save(pred_path, audio_out[idx], target_sr)
            torchaudio.save(gt_path, gt_audio[idx].unsqueeze(0), target_sr)
        records.append({
            "sample_index": idx,
            "sound_idx": sound_idx[idx],
            "pred_audio": pred_audio,
            "gt_audio": gt_audio_np,
            "pred_path": pred_path,
            "gt_path": gt_path,
        })

    if save_local:
        print(f"  Salvati {sample_count} sample audio di validazione in: {epoch_dir}")
    else:
        print(f"  Preparati {sample_count} sample audio di validazione per wandb (senza salvataggio locale)")
    return records


def build_wandb_audio_table(records, epoch: int, target_sr: int):
    if wandb is None or not records:
        return None

    table = wandb.Table(columns=["epoch", "sample_index", "sound_idx", "prediction", "ground_truth"])
    for record in records:
        table.add_data(
            epoch,
            record["sample_index"],
            record["sound_idx"],
            wandb.Audio(record["pred_audio"], sample_rate=target_sr),
            wandb.Audio(record["gt_audio"], sample_rate=target_sr),
        )
    return table


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    wandb_run = maybe_init_wandb(args)

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
        num_decoder_layers    = args.num_decoder_layers,
        nhead                 = args.nhead,
        dropout               = args.dropout,
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
        average_test_repeats = args.average_test_repeats,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )
    # ── 4. Ottimizzatore e scheduler ──────────────────────────────────────
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs // args.grad_accum
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
    )

    # ── 5. Training loop ───────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(args.num_epochs):
        model.guidance_generator.train()
        model.audio_proj.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(pbar):
            brain  = batch["brain_data"].to(device)    # [B, 6, 1024]
            audio  = batch["audio_target"].to(device)  # [B, samples]

            # a) Encode audio → latenti
            latents = encode_audio_to_latents(vae, audio, device)   # [B, C, T]

            B = latents.shape[0]

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
        current_lr = scheduler.get_last_lr()[0]
        maybe_log_wandb(wandb_run, {
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "train/lr": current_lr,
            "train/global_step": global_step,
        })

        should_eval = args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch == args.num_epochs - 1)
        if should_eval:
            val_loss = evaluate(
                model=model,
                data_loader=test_loader,
                vae=vae,
                noise_scheduler=noise_scheduler,
                device=device,
                min_snr_gamma=args.min_snr_gamma,
            )
            print(f"  Epoch {epoch+1} — val loss: {val_loss:.4f}")
            maybe_log_wandb(wandb_run, {
                "epoch": epoch + 1,
                "eval/loss": val_loss,
                "eval/best_loss": min(best_val_loss, val_loss),
                "eval/patience_counter": epochs_without_improvement,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_ckpt_path = os.path.join(args.output_dir, "best.pt")
                torch.save({
                    "epoch":             epoch + 1,
                    "model_config": create_model_config(args),
                    "guidance_generator": model.guidance_generator.state_dict(),
                    "audio_proj":        model.audio_proj.state_dict(),
                    "ip_adapter_modules": model.ip_adapter_modules.state_dict() if model.ip_adapter_modules is not None else None,
                    "trainable_backbone": model.pipe.transformer.state_dict() if args.train_backbone_cross_attention else None,
                    "optimizer":         optimizer.state_dict(),
                    "loss":              avg_loss,
                    "val_loss":          val_loss,
                    "best_epoch":        best_epoch,
                    "global_step":       global_step,
                }, best_ckpt_path)
                print(f"  Nuovo best checkpoint: {best_ckpt_path}")

                sample_records = generate_validation_samples(
                    model=model,
                    pipe=pipe,
                    dataset=test_ds,
                    device=device,
                    output_dir=args.output_dir,
                    epoch=epoch + 1,
                    target_sr=args.target_sr,
                    target_duration_s=args.target_duration_s,
                    num_samples=args.num_val_audio_samples,
                    num_inference_steps=args.val_num_inference_steps,
                    save_local=args.save_val_audio_local,
                )
                sample_table = build_wandb_audio_table(sample_records, epoch + 1, args.target_sr)
                log_payload = {
                    "epoch": epoch + 1,
                    "samples/count": len(sample_records),
                    "samples/best_epoch": best_epoch,
                }
                if sample_table is not None:
                    log_payload["samples/audio_table"] = sample_table
                maybe_log_wandb(wandb_run, log_payload)
            else:
                epochs_without_improvement += 1
                print(
                    f"  Nessun miglioramento per {epochs_without_improvement} evaluation consecutive "
                    f"(patience={args.early_stop_patience})"
                )
                maybe_log_wandb(wandb_run, {
                    "epoch": epoch + 1,
                    "eval/patience_counter": epochs_without_improvement,
                })

                if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                    print(
                        f"Early stopping attivato a epoch {epoch+1}. "
                        f"Best epoch: {best_epoch} con val loss {best_val_loss:.4f}"
                    )
                    break

    print("Training completato.")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_cv"] = args.cv
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
