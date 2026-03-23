"""
AudioNeuroDataset
=================
Adatta la struttura di nsd_topk_parcel_dataset al caso audio.

Formato dati in ingresso:
  pooled_data[roi][cv]["X_train"]   → (N, 1024)  float32  beta fMRI per ROI
  pooled_data[roi][cv]["train_sounds"] → (N,)     int      indice 1-based del suono

Dopo lo stack su roi_list:
  roi_pooled_train  → (N, num_rois, 1024)   ← questo è [B, P, V] per ParcelMapper
  train_wav_paths   → lista di N path .wav

Il __getitem__ restituisce:
  {
    "brain_data":    Tensor [num_rois, max_voxels]   (già paddato/troncato)
    "audio_target":  Tensor [samples]                 (waveform normalizzata)
    "sound_idx":     int
  }
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import re
from typing import Optional


ROI_LIST = ['HG2hem', 'PT2hem', 'PP2hem', 'mSTG2hem', 'pSTG2hem', 'aSTG2hem']


class BrainNormalizer:
    """Dataset-level z-score normalizer fit on the training split only."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, roi_data: np.ndarray, eps: float = 1e-6):
        mean = torch.from_numpy(roi_data.mean(axis=0).astype(np.float32))
        std = torch.from_numpy(roi_data.std(axis=0).astype(np.float32)).clamp_min(eps)
        return cls(mean=mean, std=std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def soundname_to_wav(sound_name: str) -> str:
    """
    Converte una stringa tipo:
    'stim145_cat04_music_exemp01'
    in:
    's2_music_1.wav'
    """
    m = re.match(r"stim\d+_cat\d+_([a-zA-Z]+)_exemp(\d+)", sound_name)
    if m is None:
        raise ValueError(f"Formato inatteso: {sound_name}")

    category = m.group(1).lower()
    exemp = int(m.group(2))

    return f"s2_{category}_{exemp}.wav"


def average_test_by_sound(X_test, test_sounds):
    """
    Media X_test sui soggetti per ciascun suono test.
    Input:
        X_test: (n_subjects * 72, n_voxels)
        test_sounds: (n_subjects * 72,)
    Output:
        X_test_avg: (72, n_voxels)
        test_sounds_avg: (72,)
    """
    unique_sounds = np.unique(test_sounds)

    X_test_avg = []
    test_sounds_avg = []

    for s in unique_sounds:
        idx = np.where(test_sounds == s)[0]
        X_test_avg.append(X_test[idx].mean(axis=0))
        test_sounds_avg.append(s)

    X_test_avg = np.vstack(X_test_avg)
    test_sounds_avg = np.array(test_sounds_avg)

    return X_test_avg, test_sounds_avg


def average_roi_data_by_sound(roi_data: np.ndarray, sound_idxs: np.ndarray) -> tuple:
    """
    Average repeated test measurements for each sound across the first axis.

    Args:
        roi_data: np.ndarray [N, num_rois, num_voxels]
        sound_idxs: np.ndarray [N,]

    Returns:
        roi_data_avg: np.ndarray [num_unique_sounds, num_rois, num_voxels]
        sound_idxs_avg: np.ndarray [num_unique_sounds,]
    """
    unique_sounds = np.unique(sound_idxs)
    roi_data_avg = []

    for sound_idx in unique_sounds:
        idx = np.where(sound_idxs == sound_idx)[0]
        roi_data_avg.append(roi_data[idx].mean(axis=0))

    roi_data_avg = np.stack(roi_data_avg, axis=0).astype(np.float32)
    return roi_data_avg, unique_sounds.astype(sound_idxs.dtype)


def load_pooled_data(
    pooled_data: dict,
    cv: str,
    roi_list: list = ROI_LIST,
    split: str = "train",
) -> tuple:
    """
    Carica e stacka i dati per tutte le ROI.

    Returns:
        roi_data   : np.ndarray  [N, num_rois, 1024]
        sound_idxs : np.ndarray  [N,]  (indici 0-based)
    """
    assert split in ("train", "test")
    key_x     = f"X_{split}"
    key_sounds = f"{split}_sounds"

    stacked = []
    sound_idxs = None

    for reg in roi_list:
        entry = pooled_data[reg][cv]
        stacked.append(entry[key_x])                     # (N, 1024)
        if sound_idxs is None:
            sound_idxs = entry[key_sounds].astype(int) - 1   # → 0-based

    roi_data = np.stack(stacked, axis=1)                 # (N, num_rois, 1024)
    return roi_data, sound_idxs


class AudioNeuroDataset(Dataset):
    """
    Dataset per brain-to-audio decoding.

    Args:
        roi_data      : np.ndarray [N, num_rois, 1024]
        sound_idxs    : np.ndarray [N,]  indici 0-based dei suoni
        wav_paths     : list[str]  percorsi ai file audio (indicizzati da sound_idxs)
        target_sr     : int        sample rate di destinazione (Stable Audio: 44100)
        target_len_s  : float      durata in secondi da estrarre / paddare
        normalize_brain : bool     z-score per campione dei dati neurali
    """

    def __init__(
        self,
        roi_data: np.ndarray,
        sound_idxs: np.ndarray,
        wav_paths: list,
        target_sr: int = 44100,
        target_len_s: float = 4.0,
        brain_normalizer: Optional[BrainNormalizer] = None,
    ):
        assert roi_data.shape[0] == len(sound_idxs), \
            f"Mismatch: roi_data {roi_data.shape[0]} vs sound_idxs {len(sound_idxs)}"

        self.roi_data       = torch.from_numpy(roi_data.astype(np.float32))  # [N, P, 1024]
        self.sound_idxs     = sound_idxs
        self.wav_paths      = wav_paths
        self.target_sr      = target_sr
        self.target_samples = int(target_sr * target_len_s)
        self.brain_normalizer = brain_normalizer

        self.num_rois    = roi_data.shape[1]   # 6
        self.max_voxels  = roi_data.shape[2]   # 1024

    def _load_audio(self, path: str) -> torch.Tensor:
        """
        Carica un file .wav, lo porta a mono, lo resampla a target_sr
        e lo porta esattamente a target_samples campioni.
        Returns: Tensor [target_samples]
        """
        wav, sr = torchaudio.load(path)

        # Stereo → mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample se necessario
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        wav = wav.squeeze(0)   # [samples]

        # Pad o tronca alla lunghezza target
        if wav.shape[0] < self.target_samples:
            pad = self.target_samples - wav.shape[0]
            wav = F.pad(wav, (0, pad))
        else:
            wav = wav[:self.target_samples]

        # Normalizza in [-1, 1]
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak

        return wav   # [target_samples]

    def __len__(self) -> int:
        return len(self.sound_idxs)

    def __getitem__(self, idx: int) -> dict:
        brain = self.roi_data[idx]           # [num_rois, 1024]

        if self.brain_normalizer is not None:
            brain = self.brain_normalizer.normalize(brain)

        sound_idx = self.sound_idxs[idx]
        audio     = self._load_audio(self.wav_paths[sound_idx])   # [target_samples]

        return {
            "brain_data":  brain,       # [6, 1024]
            "audio_target": audio,      # [target_samples]
            "sound_idx":   int(sound_idx),
        }


def build_datasets(
    pooled_data: dict,
    cv: str,
    wav_dir: str,
    sound_names: np.ndarray,
    roi_list: list = ROI_LIST,
    target_sr: int = 44100,
    target_len_s: float = 4.0,
    average_test_repeats: bool = False,
) -> tuple:
    """
    Factory function: crea train e test dataset in un colpo solo.

    Args:
        pooled_data  : dizionario raw dei dati neurali
        cv           : chiave cross-validation (es. "CV2")
        wav_dir      : cartella radice dei .wav
        sound_names  : np.ndarray dei nomi dei suoni (SoundNames.npy)

    Returns:
        train_dataset, test_dataset
    """
    all_wav_paths = [
        os.path.join(wav_dir, soundname_to_wav(n))
        for n in sound_names
    ]

    train_roi, train_idx = load_pooled_data(pooled_data, cv, roi_list, split="train")
    test_roi,  test_idx  = load_pooled_data(pooled_data, cv, roi_list, split="test")
    if average_test_repeats:
        test_roi, test_idx = average_roi_data_by_sound(test_roi, test_idx)
    brain_normalizer = BrainNormalizer.fit(train_roi)

    train_ds = AudioNeuroDataset(
        train_roi,
        train_idx,
        all_wav_paths,
        target_sr,
        target_len_s,
        brain_normalizer=brain_normalizer,
    )
    test_ds = AudioNeuroDataset(
        test_roi,
        test_idx,
        all_wav_paths,
        target_sr,
        target_len_s,
        brain_normalizer=brain_normalizer,
    )

    print(f"Train: {len(train_ds)} campioni | Test: {len(test_ds)} campioni")
    print(f"  brain_data shape: {train_ds.num_rois} ROI × {train_ds.max_voxels} voxels")
    print(f"  audio_target:     {train_ds.target_samples} samples @ {target_sr}Hz "
          f"({target_len_s}s)")

    return train_ds, test_ds
