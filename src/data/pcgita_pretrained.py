"""Extrakcia embeddings z predtrenovanych recovych modelov (wav2vec~2.0, HuBERT).

Modul tvori druhu vetvu PC-GITA pipeline: namiesto rucne navrhnutych 47
akustickych priznakov vytiahneme self-supervised reprezentacie z fronzen
encodera a agregujeme ich na uroven (pacient, domena) presne tak, ako to robi
tabular vetva v ``src/data/pcgita.py``. Vystupom je CSV s rovnakou schemou
metadat (``patient_id``, ``label``, ``domain``, ...) plus N numerickych
priznakov ``w2v2_000`` ... ``w2v2_N-1``, takze sa da priamo zoslat do toho
isteho experimentalneho protokolu (in-domain CV, pairwise prenos, multi-source)
bez akejkolvek zmeny klasifikatorov a DA implementacii.

Pristup:
    1. wav sa nacita ako mono float32, zaciatok DC odstraneny.
    2. Resample 44.1 kHz -> 16 kHz (vstupna podmienka wav2vec~2.0 / HuBERT).
    3. Audio sa rozseka na fixne segmenty (default 4 s; posledny sa doplni
       tichom). Pre wav < segment_seconds sa robi len jeden zero-padovany chunk.
    4. Cez frozen encoder sa kazdy chunk premeni na sekvenciu skrytych stavov.
       Po time-axis sa spravi mean+std pooling -> 2*H-dim utterance embedding.
    5. Cez vsetky chunky jedneho wav-u sa robi mean -> file embedding.
    6. Cez vsetky wav-y patrace tomu istemu paru (patient, domain) sa robi
       mean -> patient-domain embedding (rovnaka strategia ako v tabular
       branchi, takze vahy pacientov ostavaju vyrovnane).

Cache:
    Vystup je deterministicky pre dvojicu (model_name, segment_seconds). Cesta
    je odvodena z ``model_name`` (slash -> double-underscore), napr.
    ``data/pcgita_pretrained_facebook__wav2vec2-base_patient_domain.csv``.
    Pri opetovnom spusteni sa cache nacita; ``force=True`` ju prepise.

Vykonny model:
    Default je ``facebook/wav2vec2-base`` (95M parametrov, 768-dim). Volitelne
    je mozne zvolit ``facebook/wav2vec2-large-960h``, ``facebook/hubert-base-ls960``
    a podobne. Whisper encoder je formalne podporeny cez ``WhisperModel``, ale
    pre interface zostava klasicky ``AutoModel``-friendly default.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import DATA_DIR
from src.data.pcgita import (
    PCGITA_METADATA,
    load_pcgita_metadata,
    scan_pcgita_audio_files,
)

WAV2VEC2_SAMPLE_RATE = 16_000
DEFAULT_MODEL = 'facebook/wav2vec2-base'
DEFAULT_SEGMENT_SECONDS = 4.0


def _safe_model_slug(model_name: str) -> str:
    return model_name.replace('/', '__').replace(' ', '_')


def pretrained_cache_path(
    model_name: str = DEFAULT_MODEL,
    data_dir: str | os.PathLike[str] = DATA_DIR,
) -> Path:
    slug = _safe_model_slug(model_name)
    return Path(data_dir) / f'pcgita_pretrained_{slug}_patient_domain.csv'


def _load_wav_mono_16k(path: Path) -> np.ndarray:
    """Nacita wav ako mono float32 a prevzorkuje na 16 kHz."""
    import soundfile as sf
    import torchaudio.functional as AF

    audio, sample_rate = sf.read(str(path), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    audio = audio - float(audio.mean())
    if sample_rate != WAV2VEC2_SAMPLE_RATE:
        wave = torch.from_numpy(audio).unsqueeze(0)
        wave = AF.resample(wave, sample_rate, WAV2VEC2_SAMPLE_RATE)
        audio = wave.squeeze(0).numpy()
    return audio


def _chunk_audio(audio: np.ndarray, segment_samples: int) -> np.ndarray:
    """Rozseka 1D audio na (n_chunks, segment_samples). Posledny chunk sa zero-paduje."""
    if audio.size == 0:
        audio = np.zeros(segment_samples, dtype=np.float32)
    if audio.size < segment_samples:
        pad = segment_samples - audio.size
        return np.pad(audio, (0, pad)).reshape(1, -1).astype(np.float32, copy=False)
    n_full = audio.size // segment_samples
    leftover = audio.size - n_full * segment_samples
    full = audio[: n_full * segment_samples].reshape(n_full, segment_samples)
    if leftover > 0:
        tail = np.pad(audio[n_full * segment_samples :], (0, segment_samples - leftover))
        return np.vstack([full, tail.reshape(1, -1)]).astype(np.float32, copy=False)
    return full.astype(np.float32, copy=False)


def _load_pretrained_encoder(model_name: str, device: torch.device):
    """Lazy import transformers a vrati (feature_extractor, model, hidden_size)."""
    from transformers import AutoFeatureExtractor, AutoModel

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)
    hidden = getattr(model.config, 'hidden_size', None) or model.config.d_model
    return feature_extractor, model, int(hidden)


@torch.inference_mode()
def _encode_file(
    audio: np.ndarray,
    feature_extractor,
    model,
    device: torch.device,
    segment_samples: int,
    batch_size: int,
) -> np.ndarray:
    """Vrati 2*H-dim embedding pre jeden wav (mean+std pooling)."""
    chunks = _chunk_audio(audio, segment_samples)
    pooled = []
    for start in range(0, chunks.shape[0], batch_size):
        batch = chunks[start : start + batch_size]
        inputs = feature_extractor(
            list(batch),
            sampling_rate=WAV2VEC2_SAMPLE_RATE,
            return_tensors='pt',
            padding=True,
        )
        input_values = inputs['input_values'].to(device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state  # (B, T, H)
        if attention_mask is not None and hidden.shape[1] != attention_mask.shape[1]:
            # wav2vec2-style stride compresses time axis; fall back to mean over all
            attention_mask = None
        if attention_mask is None:
            mean = hidden.mean(dim=1)
            std = hidden.std(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp_min(1.0)
            mean = (hidden * mask).sum(dim=1) / denom
            var = ((hidden - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / denom
            std = var.clamp_min(1e-12).sqrt()
        pooled.append(torch.cat([mean, std], dim=1).cpu().numpy())
    file_embedding = np.concatenate(pooled, axis=0).mean(axis=0).astype(np.float32)
    return file_embedding


def build_pretrained_patient_domain(
    model_name: str = DEFAULT_MODEL,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    batch_size: int = 8,
    force: bool = False,
    cache_path: Path | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extrahuje embeddings z PC-GITA wav-ov a vrati patient-domain DataFrame.

    Stlpce dataframu:
        ``patient_id``, ``label``, ``domain``, ``sex``, ``age``, ``updrs``,
        ``updrs_speech``, ``hy``, ``time_after_diagnosis``, ``n_files``,
        plus ``w2v2_000`` ... ``w2v2_{2H-1}`` (mean nasledovany std cez
        time-axis a usredneny cez wav-y patrace pacientovi v danej ulohe).
    """
    if cache_path is None:
        cache_path = pretrained_cache_path(model_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        if verbose:
            print(f'[pretrained] Cache hit: {cache_path}')
        return pd.read_csv(cache_path)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not PCGITA_METADATA.exists():
        raise FileNotFoundError(
            'PC-GITA metadata not found. Cannot extract pretrained embeddings '
            f'without raw audio at {PCGITA_METADATA}.'
        )

    load_pcgita_metadata()  # validacia, ze metadata su nacitatelne
    files = scan_pcgita_audio_files()
    if files.empty:
        raise RuntimeError('PC-GITA audio scan returned 0 files.')

    if verbose:
        print(
            f'[pretrained] Loading {model_name} on {device.type.upper()} '
            f'(wav_files={len(files)}, segment={segment_seconds}s).'
        )

    feature_extractor, model, hidden_size = _load_pretrained_encoder(model_name, device)
    segment_samples = int(round(segment_seconds * WAV2VEC2_SAMPLE_RATE))

    embedding_dim = 2 * hidden_size  # mean + std
    feature_columns = [f'w2v2_{i:03d}' for i in range(embedding_dim)]

    n_total = len(files)
    file_embeddings = np.zeros((n_total, embedding_dim), dtype=np.float32)
    started = time.time()
    for idx, row in enumerate(files.itertuples(index=False)):
        try:
            audio = _load_wav_mono_16k(Path(row.path))
            file_embeddings[idx] = _encode_file(
                audio,
                feature_extractor,
                model,
                device,
                segment_samples=segment_samples,
                batch_size=batch_size,
            )
        except Exception as exc:  # noqa: BLE001 - per-file robustness, log and continue
            print(f'[pretrained] FAILED {row.path}: {type(exc).__name__}: {exc}')
        if verbose and ((idx + 1) % 200 == 0 or (idx + 1) == n_total):
            elapsed = time.time() - started
            rate = (idx + 1) / max(elapsed, 1e-6)
            eta = (n_total - idx - 1) / max(rate, 1e-6)
            print(
                f'[pretrained] {idx + 1}/{n_total} wav '
                f'({rate:.1f} wav/s, ETA {eta / 60:.1f} min)'
            )

    embeddings_df = pd.DataFrame(
        file_embeddings, columns=feature_columns, index=files.index
    )
    file_df = pd.concat([files.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)

    group_cols = [
        'patient_id',
        'label',
        'domain',
        'sex',
        'age',
        'updrs',
        'updrs_speech',
        'hy',
        'time_after_diagnosis',
    ]
    grouped = file_df.groupby(group_cols, dropna=False)
    n_files_series = grouped.size()
    patient_df = grouped[feature_columns].mean().reset_index()
    patient_df['n_files'] = n_files_series.to_numpy()
    patient_df.to_csv(cache_path, index=False)
    if verbose:
        print(
            f'[pretrained] Saved patient-domain embeddings: {cache_path} '
            f'(rows={len(patient_df)}, dim={embedding_dim})'
        )
    return patient_df


__all__ = [
    'DEFAULT_MODEL',
    'DEFAULT_SEGMENT_SECONDS',
    'WAV2VEC2_SAMPLE_RATE',
    'build_pretrained_patient_domain',
    'pretrained_cache_path',
]
