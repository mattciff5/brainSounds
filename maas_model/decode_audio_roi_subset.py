"""
decode_audio_roi_subset.py
==========================

Genera audio a partire da dati neurali usando un checkpoint addestrato,
permettendo di scegliere quali ROI mantenere attive durante la ricostruzione.

Uso tipico:
    python decode_audio_roi_subset.py \
        --data_path /path/pooled_data.pkl \
        --wav_dir /path/wav \
        --sound_names /path/SoundNames.npy \
        --ckpt_path checkpoints/epoch_60.pt \
        --output_dir outputs/roi_subset \
        --split test \
        --average_test_repeats \
        --conditioning_mode empty_prompt_ip_adapter \
        --num_inference_steps 100 \
        --guidance_scale 3.0 \
        --keep_rois HG2hem,PP2hem,aSTG2hem \
        --samples 0-11

Note:
- Il modello e stato addestrato con tutte le ROI. Qui si fa un'ablazione in test-time:
  le ROI escluse vengono azzerate prima della ricostruzione.
- Questo e utile per ascoltare quanto cambia l'audio quando mantieni solo certe aree.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from diffusers import StableAudioPipeline
from diffusers.pipelines.stable_audio.pipeline_stable_audio import get_1d_rotary_pos_embed

from dataset import ROI_LIST, build_datasets
from model import AudioNeuroAdapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--wav_dir", type=str, required=True)
    p.add_argument("--sound_names", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./decoded_audio_roi_subset")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--cv", type=str, default="CV2")
    p.add_argument("--stable_audio_id", type=str, default="stabilityai/stable-audio-open-1.0")
    p.add_argument("--num_decoder_queries", type=int, default=16)
    p.add_argument("--target_duration_s", type=float, default=1.0)
    p.add_argument("--target_sr", type=int, default=44100)
    p.add_argument(
        "--conditioning_mode",
        type=str,
        default="brain_only",
        choices=["brain_only", "empty_prompt_plus_brain", "empty_prompt_ip_adapter"],
    )
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--average_test_repeats", action="store_true")
    p.add_argument("--save_groundtruth", action="store_true")
    p.add_argument(
        "--keep_rois",
        type=str,
        default=None,
        help="ROI da mantenere, separate da virgola. Esempio: HG2hem,PP2hem,aSTG2hem",
    )
    p.add_argument(
        "--drop_rois",
        type=str,
        default=None,
        help="ROI da azzerare, separate da virgola. Esempio: PT2hem,pSTG2hem",
    )
    p.add_argument(
        "--samples",
        type=str,
        default="all",
        help="Selezione sample 0-based. Esempi: all | 0-11 | 0,5,8 | 0-11,24-35",
    )
    p.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Offset opzionale per il seed dei sample selezionati.",
    )
    return p.parse_args()


def parse_roi_list(text: str | None) -> list[str]:
    if text is None:
        return []
    rois = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [roi for roi in rois if roi not in ROI_LIST]
    if invalid:
        raise ValueError(f"ROI non valide: {invalid}. ROI disponibili: {ROI_LIST}")
    return rois


def compute_active_roi_mask(keep_rois: list[str], drop_rois: list[str]) -> torch.Tensor:
    if keep_rois and drop_rois:
        raise ValueError("Specifica solo una tra --keep_rois e --drop_rois.")

    active = torch.ones(len(ROI_LIST), dtype=torch.float32)

    if keep_rois:
        active.zero_()
        for roi in keep_rois:
            active[ROI_LIST.index(roi)] = 1.0
    elif drop_rois:
        for roi in drop_rois:
            active[ROI_LIST.index(roi)] = 0.0

    return active


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


def selection_tag(active_mask: torch.Tensor) -> str:
    active_rois = [roi for roi, is_on in zip(ROI_LIST, active_mask.tolist()) if is_on > 0.5]
    if len(active_rois) == len(ROI_LIST):
        return "all_rois"
    if not active_rois:
        return "no_rois"
    return "keep_" + "_".join(active_rois)


@torch.no_grad()
def decode(args):
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    keep_rois = parse_roi_list(args.keep_rois)
    drop_rois = parse_roi_list(args.drop_rois)
    active_roi_mask = compute_active_roi_mask(keep_rois, drop_rois)
    active_roi_mask_device = active_roi_mask.view(1, -1, 1).to(device)
    tag = selection_tag(active_roi_mask)

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
    pipe = StableAudioPipeline.from_pretrained(stable_audio_id, torch_dtype=torch.float32).to(device)

    model = AudioNeuroAdapter(
        pipe=pipe,
        num_rois=len(ROI_LIST),
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
        model.ip_adapter_scale.data.copy_(ckpt["ip_adapter_scale"].to(device=device, dtype=model.ip_adapter_scale.dtype))
    if ckpt.get("brain_prompt_scale") is not None:
        model.brain_prompt_scale.data.copy_(ckpt["brain_prompt_scale"].to(device=device, dtype=model.brain_prompt_scale.dtype))
    if model.ip_adapter_modules is not None and ckpt.get("ip_adapter_modules") is not None:
        model.ip_adapter_modules.load_state_dict(ckpt["ip_adapter_modules"])
    if train_backbone_cross_attention and ckpt.get("trainable_backbone") is not None:
        model.pipe.transformer.load_state_dict(ckpt["trainable_backbone"])
    if ckpt.get("_brain_norm_mean") is not None:
        model._brain_norm_mean = ckpt["_brain_norm_mean"].to(device)
        model._brain_norm_std = ckpt["_brain_norm_std"].to(device)
    model.eval()

    print(f"Checkpoint caricato: {args.ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    print(f"ROI attive: {[roi for roi, on in zip(ROI_LIST, active_roi_mask.tolist()) if on > 0.5]}")

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

    selected_indices = parse_sample_selection(args.samples, len(dataset))
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    sample_counter = 0
    for batch in tqdm(loader, desc="Decoding ROI subset"):
        brain = batch["brain_data"].to(device)
        sound_idx = batch["sound_idx"]
        batch_size = brain.shape[0]

        # Applica la maschera ROI: le ROI escluse vengono azzerate.
        brain = brain * active_roi_mask_device

        text_audio_duration_embeds, audio_duration_embeds = model.build_conditioning(
            brain_data=brain,
            device=device,
        )

        uncond_embeds = None
        uncond_audio_duration_embeds = None
        if args.guidance_scale != 1.0:
            uncond_embeds, uncond_audio_duration_embeds = model.build_unconditional_conditioning(
                batch_size=batch_size,
                device=device,
                dtype=text_audio_duration_embeds.dtype,
            )

        scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
        scheduler.set_timesteps(args.num_inference_steps, device=device)

        waveform_length = int(pipe.transformer.config.sample_size)
        latent_shape = (batch_size, pipe.transformer.config.in_channels, waveform_length)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed_offset + sample_counter)
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float32)
        latents = latents * scheduler.init_noise_sigma

        rotary_embedding = get_1d_rotary_pos_embed(
            pipe.rotary_embed_dim,
            latents.shape[2] + audio_duration_embeds.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        rotary_embedding = tuple(r.to(device) for r in rotary_embedding)

        for t in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, t)

            noise_pred_cond = model.pipe.transformer(
                hidden_states=latent_input,
                timestep=t.unsqueeze(0).expand(batch_size),
                encoder_hidden_states=text_audio_duration_embeds,
                global_hidden_states=audio_duration_embeds,
                rotary_embedding=rotary_embedding,
                return_dict=False,
            )[0]

            if args.guidance_scale == 1.0:
                noise_pred = noise_pred_cond
            else:
                noise_pred_uncond = model.pipe.transformer(
                    hidden_states=latent_input,
                    timestep=t.unsqueeze(0).expand(batch_size),
                    encoder_hidden_states=uncond_embeds,
                    global_hidden_states=uncond_audio_duration_embeds,
                    rotary_embedding=rotary_embedding,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        waveform_end = int(target_duration_s * pipe.vae.config.sampling_rate)
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
        audio_out = pipe.vae.decode(latents / scaling_factor).sample
        audio_out = audio_out[:, :, :waveform_end]
        audio_out = audio_out.mean(dim=1, keepdim=True).cpu()

        for i in range(batch_size):
            wav = audio_out[i]
            s_idx = int(sound_idx[i])
            original_sample_idx = selected_indices[sample_counter]
            out_name = f"pred_{tag}_sound{s_idx:04d}_sample{original_sample_idx:04d}.wav"
            out_path = Path(args.output_dir) / out_name
            torchaudio.save(str(out_path), wav, target_sr)

            if args.save_groundtruth:
                gt_wav = batch["audio_target"][i].unsqueeze(0)
                gt_name = f"gt_sound{s_idx:04d}_sample{original_sample_idx:04d}.wav"
                torchaudio.save(str(Path(args.output_dir) / gt_name), gt_wav, target_sr)

            sample_counter += 1

    print(f"\nDecodifica completata. Salvati {sample_counter} file in: {args.output_dir}")


if __name__ == "__main__":
    decode(parse_args())
