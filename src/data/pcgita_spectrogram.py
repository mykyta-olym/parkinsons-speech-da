"""
Log-mel spektrogramy pre PC-GITA.

Táto vetva používa priamo audio signál a vytvára fixne dlhé log-mel
spektrogramy. Tréning môže používať viac súborov jedného pacienta, ale split
aj finálna evaluácia musia byť patient-wise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from src.config import DATA_DIR
from src.data.pcgita import _load_wav_mono, scan_pcgita_audio_files

PCGITA_LOGMEL_CACHE = Path(DATA_DIR) / 'pcgita_logmel_4s_64mels.pt'


class PCGitaLogMelDataset(Dataset):
    """Dataset nad predpočítanými log-mel spektrogramami."""

    def __init__(self, cache: dict, indices: np.ndarray):
        self.specs = cache['specs']
        self.labels = torch.as_tensor(cache['labels'], dtype=torch.long)
        self.patient_ids = np.asarray(cache['patient_ids'])
        self.domains = np.asarray(cache['domains'])
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        return self.specs[idx].float(), self.labels[idx], self.patient_ids[idx]


def _fix_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """Center crop alebo zero-pad na fixnú dĺžku."""
    n = waveform.shape[-1]
    if n == target_len:
        return waveform
    if n > target_len:
        start = (n - target_len) // 2
        return waveform[..., start:start + target_len]
    pad_left = (target_len - n) // 2
    pad_right = target_len - n - pad_left
    return torch.nn.functional.pad(waveform, (pad_left, pad_right))


def _audio_to_logmel(
    path: str,
    sample_rate: int,
    seconds: float,
    n_mels: int,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    db_transform: torchaudio.transforms.AmplitudeToDB,
) -> torch.Tensor:
    audio, sr = _load_wav_mono(Path(path))
    waveform = torch.from_numpy(audio).unsqueeze(0)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    waveform = waveform - waveform.mean()
    peak = waveform.abs().max().clamp_min(1e-6)
    waveform = waveform / peak
    waveform = _fix_length(waveform, int(sample_rate * seconds))

    mel = mel_transform(waveform)
    logmel = db_transform(mel)
    logmel = (logmel - logmel.mean()) / logmel.std().clamp_min(1e-6)
    return logmel.to(dtype=torch.float16)


def build_logmel_cache(
    force: bool = False,
    sample_rate: int = 16_000,
    seconds: float = 4.0,
    n_mels: int = 64,
) -> dict:
    """
    Vytvorí alebo načíta cache log-mel spektrogramov.

    Cache obsahuje iba oficiálnych 100 PC-GITA subjektov z metadát. Extra ID
    bez metadát sa nepoužívajú.
    """
    if PCGITA_LOGMEL_CACHE.exists() and not force:
        return torch.load(PCGITA_LOGMEL_CACHE, map_location='cpu', weights_only=False)

    records = scan_pcgita_audio_files()
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        win_length=400,
        hop_length=160,
        f_min=50.0,
        f_max=sample_rate / 2,
        n_mels=n_mels,
        power=2.0,
        normalized=False,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    specs = []
    total = len(records)
    for idx, row in records.iterrows():
        specs.append(
            _audio_to_logmel(
                row['path'],
                sample_rate=sample_rate,
                seconds=seconds,
                n_mels=n_mels,
                mel_transform=mel_transform,
                db_transform=db_transform,
            )
        )
        if (idx + 1) % 250 == 0 or (idx + 1) == total:
            print(f'  PC-GITA logmel: {idx + 1}/{total} wav files')

    cache = {
        'specs': torch.stack(specs, dim=0),
        'labels': records['label'].to_numpy(dtype=np.int64),
        'patient_ids': records['patient_id'].to_numpy(),
        'domains': records['domain'].to_numpy(),
        'paths': records['path'].to_numpy(),
        'sample_rate': sample_rate,
        'seconds': seconds,
        'n_mels': n_mels,
    }
    torch.save(cache, PCGITA_LOGMEL_CACHE)
    return cache


def spectrogram_indices(
    cache: dict,
    domains: list[str],
    patient_ids: set[str] | None = None,
) -> np.ndarray:
    """Indexy podľa domény a voliteľne podľa zoznamu pacientov."""
    cache_domains = np.asarray(cache['domains'])
    cache_patients = np.asarray(cache['patient_ids'])
    mask = np.isin(cache_domains, domains)
    if patient_ids is not None:
        mask &= np.isin(cache_patients, list(patient_ids))
    return np.where(mask)[0]
