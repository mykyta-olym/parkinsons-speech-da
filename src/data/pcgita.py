"""Nacitanie PC-GITA a vypocet jednoduchych audio priznakov.

PC-GITA tu berieme priamo z wav suborov. Najprv sa nacitaju oficialne
metadata, potom sa najdu nahravky pre tychto pacientov a nakoniec sa ulozi
cache, aby sa priznaky nemuseli pocitat pri kazdom spusteni.
"""

from __future__ import annotations

import math
import re
import wave
from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.fftpack import dct

from src.config import DATA_DIR

PCGITA_DIR = Path(DATA_DIR) / 'PC-GITA' / 'PC-GITA_per_task_44100Hz'
PCGITA_METADATA = PCGITA_DIR / 'Copia de PCGITA_metadata.xlsx'
PCGITA_FILE_FEATURES = Path(DATA_DIR) / 'pcgita_features_file.csv'
PCGITA_PATIENT_DOMAIN_FEATURES = Path(DATA_DIR) / 'pcgita_features_patient_domain.csv'

PATIENT_RE = re.compile(r'(AVPEPUDEAC\d{4}|AVPEPUDEA\d{4})', re.IGNORECASE)

DOMAIN_MAP = {
    'modulated vowels': 'vowels',
    'ddk analysis': 'ddk',
    'words': 'words',
    'sentences': 'sentences',
    'sentences2': 'sentences2',
    'read text': 'read_text',
    'monologue': 'monologue',
}

META_COLUMNS = {
    'path',
    'patient_id',
    'label',
    'domain',
    'task_detail',
    'sex',
    'age',
    'updrs',
    'updrs_speech',
    'hy',
    'time_after_diagnosis',
    'n_files',
}


def _xlsx_first_sheet_rows(path: Path) -> list[list[str]]:
    """Male citanie prveho XLSX sheetu bez openpyxl."""
    ns = {
        'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    }

    def col_index(cell_ref: str) -> int:
        letters = re.match(r'([A-Z]+)', cell_ref).group(1)
        idx = 0
        for ch in letters:
            idx = idx * 26 + ord(ch) - ord('A') + 1
        return idx - 1

    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            for item in root.findall('a:si', ns):
                shared_strings.append(
                    ''.join(text.text or '' for text in item.findall('.//a:t', ns))
                )

        def cell_value(cell: ET.Element) -> str:
            kind = cell.attrib.get('t')
            value = cell.find('a:v', ns)
            if value is None:
                return ''.join(text.text or '' for text in cell.findall('.//a:t', ns))
            raw = value.text or ''
            if kind == 's' and raw:
                return shared_strings[int(raw)]
            return raw

        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        rows: list[list[str]] = []
        for row in sheet.findall('a:sheetData/a:row', ns):
            values: list[str] = []
            for cell in row.findall('a:c', ns):
                idx = col_index(cell.attrib['r'])
                while len(values) <= idx:
                    values.append('')
                values[idx] = cell_value(cell)
            if any(values):
                rows.append(values)
    return rows


def load_pcgita_metadata(path: Path = PCGITA_METADATA) -> pd.DataFrame:
    """Nacita metadata oficialnych PC-GITA subjektov."""
    rows = _xlsx_first_sheet_rows(path)
    if not rows:
        raise FileNotFoundError(f'Metadáta PC-GITA sa nepodarilo načítať: {path}')

    header = [h.strip().lower() for h in rows[0]]
    records = []
    for row in rows[1:]:
        padded = row + [''] * (len(header) - len(row))
        raw = dict(zip(header, padded, strict=False))
        patient_id = raw.get('recoding original name', '').strip().upper()
        if not patient_id:
            continue
        is_control = patient_id.startswith('AVPEPUDEAC')
        records.append({
            'patient_id': patient_id,
            'label': 0 if is_control else 1,
            'sex': raw.get('sex', ''),
            'age': _safe_float(raw.get('age', '')),
            'updrs': _safe_float(raw.get('updrs', '')),
            'updrs_speech': _safe_float(raw.get('updrs-speech', '')),
            'hy': _safe_float(raw.get('h/y', '')),
            'time_after_diagnosis': _safe_float(raw.get('time after diagnosis', '')),
        })
    return pd.DataFrame(records)


def _safe_float(value: str) -> float:
    try:
        if value == '' or value is None:
            return float('nan')
        return float(value)
    except ValueError:
        return float('nan')


def _patient_id_from_path(path: Path) -> str | None:
    match = PATIENT_RE.search(path.name)
    return match.group(1).upper() if match else None


def _domain_from_path(path: Path, root: Path = PCGITA_DIR) -> tuple[str, str]:
    rel_parts = path.relative_to(root).parts
    top = rel_parts[0].lower()
    domain = DOMAIN_MAP.get(top, top.replace(' ', '_'))
    task_detail = rel_parts[1] if len(rel_parts) > 1 else domain
    return domain, task_detail


def scan_pcgita_audio_files(root: Path = PCGITA_DIR) -> pd.DataFrame:
    """
    Najde wav subory PC-GITA a ponecha iba subjekty z oficialnych metadat.

    V lokalnej kopii su aj extra ID bez metadat. Tie nepouzivame, aby sa
    nemiesali neoverene subjekty s oficialnym zoznamom.
    """
    metadata = load_pcgita_metadata()
    meta_by_id = metadata.set_index('patient_id').to_dict(orient='index')
    official_ids = set(meta_by_id)

    records = []
    for wav_path in sorted(root.rglob('*.wav')):
        patient_id = _patient_id_from_path(wav_path)
        if patient_id not in official_ids:
            continue
        domain, task_detail = _domain_from_path(wav_path, root)
        meta = meta_by_id[patient_id]
        records.append({
            'path': str(wav_path),
            'patient_id': patient_id,
            'label': int(meta['label']),
            'domain': domain,
            'task_detail': task_detail,
            'sex': meta['sex'],
            'age': meta['age'],
            'updrs': meta['updrs'],
            'updrs_speech': meta['updrs_speech'],
            'hy': meta['hy'],
            'time_after_diagnosis': meta['time_after_diagnosis'],
        })
    return pd.DataFrame(records)


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f'Nepodporovaná šírka vzorky {sample_width}: {path}')

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    audio = audio - float(np.mean(audio))
    return audio.astype(np.float32), sample_rate


def _frame_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_len = int(round(0.025 * sample_rate))
    hop = int(round(0.010 * sample_rate))
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))
    n_frames = 1 + math.floor((len(audio) - frame_len) / hop)
    shape = (n_frames, frame_len)
    strides = (audio.strides[0] * hop, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    return np.array(frames, copy=True)


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    low_mel = _hz_to_mel(50.0)
    high_mel = _hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            filters[m - 1, left:center] = (
                np.arange(left, center) - left
            ) / max(center - left, 1)
        if right > center:
            filters[m - 1, center:right] = (
                right - np.arange(center, right)
            ) / max(right - center, 1)
    return filters


def _stat_features(values: Iterable[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f'{prefix}_mean': 0.0, f'{prefix}_std': 0.0}
    return {
        f'{prefix}_mean': float(np.mean(arr)),
        f'{prefix}_std': float(np.std(arr)),
    }


def extract_audio_features(path: str | Path) -> dict[str, float]:
    """
    Extrahuje kompaktné akustické príznaky z jedného wav súboru.

    Príznaky sú zámerne všeobecné: MFCC štatistiky, spektrálne štatistiky,
    energia, ZCR a jednoduchý odhad dominantnej frekvencie v pásme hlasu.
    Nepoužívame cudzie hotové PC-GITA príznaky, aby bol postup jednotný.
    """
    audio, sample_rate = _load_wav_mono(Path(path))
    duration = len(audio) / sample_rate
    eps = 1e-10

    frames = _frame_audio(audio, sample_rate)
    window = np.hamming(frames.shape[1]).astype(np.float32)
    frames_win = frames * window

    n_fft = 1
    while n_fft < frames.shape[1]:
        n_fft *= 2
    spectrum = np.abs(np.fft.rfft(frames_win, n=n_fft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    power_sum = np.sum(spectrum, axis=1) + eps

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + eps)
    zcr = np.mean(frames[:, 1:] * frames[:, :-1] < 0, axis=1)
    centroid = np.sum(spectrum * freqs[None, :], axis=1) / power_sum
    bandwidth = np.sqrt(
        np.sum(spectrum * (freqs[None, :] - centroid[:, None]) ** 2, axis=1)
        / power_sum
    )

    cumulative = np.cumsum(spectrum, axis=1)
    rolloff_idx = np.argmax(cumulative >= 0.85 * power_sum[:, None], axis=1)
    rolloff = freqs[rolloff_idx]

    flatness = np.exp(np.mean(np.log(spectrum + eps), axis=1)) / (
        np.mean(spectrum + eps, axis=1)
    )
    prob = spectrum / power_sum[:, None]
    entropy = -np.sum(prob * np.log(prob + eps), axis=1) / math.log(spectrum.shape[1])

    norm_spec = spectrum / power_sum[:, None]
    if len(norm_spec) > 1:
        flux = np.sqrt(np.mean(np.diff(norm_spec, axis=0) ** 2, axis=1))
    else:
        flux = np.array([0.0])

    voice_band = (freqs >= 50.0) & (freqs <= 500.0)
    voice_power = spectrum[:, voice_band]
    voice_freqs = freqs[voice_band]
    peak_freq = voice_freqs[np.argmax(voice_power, axis=1)]

    rms_threshold = max(float(np.percentile(rms, 20)), 1e-4)
    voiced_mask = rms > rms_threshold
    silence_ratio = float(np.mean(rms <= rms_threshold))

    mel_filters = _mel_filterbank(sample_rate, n_fft, n_mels=26)
    mel_energy = np.maximum(spectrum @ mel_filters.T, eps)
    mfcc = dct(np.log(mel_energy), type=2, axis=1, norm='ortho')[:, :13]

    features = {
        'duration': float(duration),
        'voiced_ratio': float(np.mean(voiced_mask)),
        'silence_ratio': silence_ratio,
    }
    for name, values in [
        ('rms', rms),
        ('zcr', zcr),
        ('centroid', centroid),
        ('bandwidth', bandwidth),
        ('rolloff', rolloff),
        ('flatness', flatness),
        ('entropy', entropy),
        ('flux', flux),
        ('peak_freq', peak_freq[voiced_mask] if np.any(voiced_mask) else peak_freq),
    ]:
        features.update(_stat_features(values, name))

    for idx in range(mfcc.shape[1]):
        coeff = mfcc[:, idx]
        features[f'mfcc_{idx:02d}_mean'] = float(np.mean(coeff))
        features[f'mfcc_{idx:02d}_std'] = float(np.std(coeff))

    return features


def build_pcgita_feature_cache(force: bool = False) -> pd.DataFrame:
    """
    Vytvorí alebo načíta PC-GITA feature cache.

    Súborová cache je užitočná na audit. Hlavný experiment používa agregáciu
    pacient-doména, aby pacient s 45 slovami nemal 45-krát väčšiu váhu.
    """
    if PCGITA_PATIENT_DOMAIN_FEATURES.exists() and not force:
        return pd.read_csv(PCGITA_PATIENT_DOMAIN_FEATURES)

    if PCGITA_FILE_FEATURES.exists() and not force:
        file_df = pd.read_csv(PCGITA_FILE_FEATURES)
    else:
        records = scan_pcgita_audio_files()
        rows = []
        total = len(records)
        for idx, row in records.iterrows():
            feats = extract_audio_features(row['path'])
            rows.append({**row.to_dict(), **feats})
            if (idx + 1) % 250 == 0 or (idx + 1) == total:
                print(f'  PC-GITA features: {idx + 1}/{total} wav files')
        file_df = pd.DataFrame(rows)
        file_df.to_csv(PCGITA_FILE_FEATURES, index=False)

    feature_cols = get_feature_columns(file_df)
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
    patient_df = grouped[feature_cols].mean().reset_index()
    patient_df['n_files'] = grouped.size().to_numpy()
    patient_df.to_csv(PCGITA_PATIENT_DOMAIN_FEATURES, index=False)
    return patient_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Vráti len modelové akustické príznaky, bez metadát."""
    return [
        col for col in df.columns
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]


def domain_matrix(domain: str, df: pd.DataFrame | None = None):
    """Vráti X, y, patient_ids a názvy príznakov pre jednu PC-GITA doménu."""
    data = build_pcgita_feature_cache() if df is None else df
    subset = data[data['domain'] == domain].copy()
    if subset.empty:
        available = ', '.join(sorted(data['domain'].unique()))
        raise ValueError(f'Neznáma doména {domain!r}. Dostupné: {available}')
    feature_cols = get_feature_columns(subset)
    return (
        subset[feature_cols].to_numpy(dtype=np.float32),
        subset['label'].to_numpy(dtype=np.int64),
        subset['patient_id'].to_numpy(),
        feature_cols,
    )
