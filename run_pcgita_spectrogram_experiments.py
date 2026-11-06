"""PC-GITA CNN experimenty nad log-mel spektrogramami.

CNN sa trenuje na jednotlivych suboroch, lebo tak vidi viac prikladov.
Vyhodnotenie je patient-wise: skore vsetkych target suborov pacienta sa
spriemeruje a az potom sa pocitaju metriky.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from src.config import DATA_DIR, DEVICE, SEEDS
from src.data.pcgita_spectrogram import (
    PCGitaLogMelDataset,
    build_logmel_cache,
    spectrogram_indices,
)
from src.models import SpectrogramCNN, SpectrogramDANN
from src.models.domain_adaptation import coral_loss, mmd_loss
from src.utils import set_seed

DEFAULT_PAIRS = [
    ('multi', 'words'),
    ('multi', 'sentences2'),
    ('multi', 'read_text'),
    ('multi', 'monologue'),
    ('sentences2', 'words'),
    ('words', 'sentences2'),
    ('sentences', 'read_text'),
    ('read_text', 'sentences'),
]

DEFAULT_METHODS = ['cnn_source', 'cnn_dann', 'cnn_mmd', 'cnn_coral']


@dataclass
class AudioSplit:
    source_idx: np.ndarray
    target_train_idx: np.ndarray
    target_test_idx: np.ndarray
    train_patients: set[str]
    test_patients: set[str]


def _parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(',') if part.strip()]


def _parse_pairs(value: str) -> list[tuple[str, str]]:
    pairs = []
    for item in _parse_list(value):
        source, target = item.split('->', 1)
        pairs.append((source.strip(), target.strip()))
    return pairs


def _patient_labels(cache: dict, domain: str) -> tuple[np.ndarray, np.ndarray]:
    domains = np.asarray(cache['domains'])
    patients = np.asarray(cache['patient_ids'])
    labels = np.asarray(cache['labels'])
    idx = np.where(domains == domain)[0]
    patient_ids = sorted(set(patients[idx]))
    y = []
    for pid in patient_ids:
        patient_file_labels = labels[idx[patients[idx] == pid]]
        y.append(int(np.bincount(patient_file_labels).argmax()))
    return np.asarray(patient_ids), np.asarray(y, dtype=np.int64)


def _source_domains(source: str, target: str, all_domains: list[str]) -> list[str]:
    if source == 'multi':
        return [domain for domain in all_domains if domain != target]
    return [part.strip() for part in source.split('+') if part.strip()]


def _make_split(
    cache: dict,
    source_domains: list[str],
    target: str,
    split_seed: int,
    target_test_size: float,
) -> AudioSplit:
    patient_ids, patient_y = _patient_labels(cache, target)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=target_test_size,
        random_state=split_seed,
    )
    train_idx, test_idx = next(splitter.split(patient_ids, patient_y, groups=patient_ids))
    train_patients = set(patient_ids[train_idx])
    test_patients = set(patient_ids[test_idx])

    return AudioSplit(
        source_idx=spectrogram_indices(cache, source_domains, train_patients),
        target_train_idx=spectrogram_indices(cache, [target], train_patients),
        target_test_idx=spectrogram_indices(cache, [target], test_patients),
        train_patients=train_patients,
        test_patients=test_patients,
    )


def _class_weights(cache: dict, indices: np.ndarray) -> torch.Tensor:
    labels = np.asarray(cache['labels'])[indices]
    classes = np.unique(labels)
    weights = []
    for cls in classes:
        weights.append(len(labels) / (len(classes) * np.sum(labels == cls)))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def _loaders(cache: dict, split: AudioSplit, seed: int, batch_size: int):
    generator = torch.Generator().manual_seed(seed)
    source_loader = DataLoader(
        PCGitaLogMelDataset(cache, split.source_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    target_loader = DataLoader(
        PCGitaLogMelDataset(cache, split.target_train_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    test_loader = DataLoader(
        PCGitaLogMelDataset(cache, split.target_test_idx),
        batch_size=batch_size,
        shuffle=False,
    )
    return source_loader, target_loader, test_loader


def _paired_batches(source_loader, target_loader):
    if len(source_loader) >= len(target_loader):
        return zip(source_loader, cycle(target_loader)), len(source_loader)
    return zip(cycle(source_loader), target_loader), len(target_loader)


def _spec_augment(x: torch.Tensor) -> torch.Tensor:
    """Jednoduchy SpecAugment: male masky frekvencie a casu."""
    if x.size(0) == 0:
        return x
    x = x.clone()
    _, _, n_freq, n_time = x.shape
    for item in range(x.size(0)):
        freq_width = int(torch.randint(0, min(8, n_freq) + 1, (1,), device=x.device))
        if freq_width > 0:
            start = int(torch.randint(0, n_freq - freq_width + 1, (1,), device=x.device))
            x[item, :, start:start + freq_width, :] = 0
        time_width = int(torch.randint(0, min(32, n_time) + 1, (1,), device=x.device))
        if time_width > 0:
            start = int(torch.randint(0, n_time - time_width + 1, (1,), device=x.device))
            x[item, :, :, start:start + time_width] = 0
    return x


def _train_source(model, loader, epochs: int, lr: float, class_weights: torch.Tensor):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x, y, _ in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            x = _spec_augment(x)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f'    epoch {epoch + 1:03d}/{epochs} loss={total / max(len(loader), 1):.4f}')


def _train_dann(model, source_loader, target_loader, epochs: int, lr: float,
                class_weights: torch.Tensor):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        model.grl.lambda_val = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        total = 0.0
        paired, n_batches = _paired_batches(source_loader, target_loader)
        for (sx, sy, _), (tx, _, _) in paired:
            sx, sy, tx = sx.to(DEVICE), sy.to(DEVICE), tx.to(DEVICE)
            sx = _spec_augment(sx)
            tx = _spec_augment(tx)
            n_s = sx.size(0)
            x = torch.cat([sx, tx], dim=0)
            class_logits, domain_logits, _ = model(x)
            class_loss = criterion(class_logits[:n_s], sy)
            domain_labels = torch.cat([
                torch.zeros(n_s, dtype=torch.long, device=DEVICE),
                torch.ones(tx.size(0), dtype=torch.long, device=DEVICE),
            ])
            domain_loss = domain_criterion(domain_logits, domain_labels)
            loss = class_loss + domain_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f'    epoch {epoch + 1:03d}/{epochs} loss={total / max(n_batches, 1):.4f}')


def _train_moment(model, source_loader, target_loader, epochs: int, lr: float,
                  class_weights: torch.Tensor, method: str, da_lambda: float):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        paired, n_batches = _paired_batches(source_loader, target_loader)
        for (sx, sy, _), (tx, _, _) in paired:
            sx, sy, tx = sx.to(DEVICE), sy.to(DEVICE), tx.to(DEVICE)
            sx = _spec_augment(sx)
            tx = _spec_augment(tx)
            n_s = sx.size(0)
            x = torch.cat([sx, tx], dim=0)
            logits, features = model(x)
            source_logits = logits[:n_s]
            source_features = features[:n_s]
            target_features = features[n_s:]
            class_loss = criterion(source_logits, sy)
            if method == 'cnn_mmd':
                da_loss = mmd_loss(source_features, target_features)
            else:
                da_loss = coral_loss(source_features, target_features)
            loss = class_loss + da_lambda * da_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f'    epoch {epoch + 1:03d}/{epochs} loss={total / max(n_batches, 1):.4f}')


def _patient_scores(model, loader, method: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    per_patient: dict[str, list[float]] = {}
    labels: dict[str, int] = {}
    with torch.no_grad():
        for x, y, patient_ids in loader:
            x = x.to(DEVICE)
            if method == 'cnn_dann':
                logits, _, _ = model(x)
            else:
                logits, _ = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for pid, label, prob in zip(patient_ids, y.numpy(), probs, strict=True):
                per_patient.setdefault(str(pid), []).append(float(prob))
                labels[str(pid)] = int(label)
    patient_ids = sorted(per_patient)
    y_true = np.asarray([labels[pid] for pid in patient_ids], dtype=np.int64)
    y_score = np.asarray([np.mean(per_patient[pid]) for pid in patient_ids], dtype=np.float32)
    return y_true, y_score


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'auc': roc_auc_score(y_true, y_score),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
    }


def _run_method(cache: dict, split: AudioSplit, method: str, seed: int,
                epochs: int, batch_size: int, lr: float) -> dict:
    set_seed(seed)
    source_loader, target_loader, test_loader = _loaders(cache, split, seed, batch_size)
    weights = _class_weights(cache, split.source_idx)

    if method == 'cnn_dann':
        model = SpectrogramDANN().to(DEVICE)
        _train_dann(model, source_loader, target_loader, epochs, lr, weights)
    else:
        model = SpectrogramCNN().to(DEVICE)
        if method == 'cnn_source':
            _train_source(model, source_loader, epochs, lr, weights)
        elif method in {'cnn_mmd', 'cnn_coral'}:
            da_lambda = 0.5 if method == 'cnn_mmd' else 10.0
            _train_moment(model, source_loader, target_loader, epochs, lr, weights,
                          method=method, da_lambda=da_lambda)
        else:
            raise ValueError(f'Unknown method: {method}')

    y_true, y_score = _patient_scores(model, test_loader, method)
    return _metrics(y_true, y_score)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ['auc', 'f1', 'balanced_accuracy', 'sensitivity', 'specificity']
    summary = (
        df.groupby(['source', 'target', 'method'])[metric_cols]
        .agg(['mean', 'std', 'min', 'max'])
        .reset_index()
    )
    summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary.sort_values(['target', 'auc_mean'], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser(description='PC-GITA log-mel CNN experiments')
    parser.add_argument('--force-logmel', action='store_true')
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--pairs', default=','.join(f'{s}->{t}' for s, t in DEFAULT_PAIRS))
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in SEEDS[:2]))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--target-test-size', type=float, default=0.4)
    parser.add_argument('--split-seed', type=int, default=12345)
    args = parser.parse_args()

    start = time.time()
    print(f'Device: {DEVICE}')
    cache = build_logmel_cache(force=args.force_logmel)
    print(f"Logmel cache: specs={tuple(cache['specs'].shape)}, domains={len(set(cache['domains']))}")
    all_domains = sorted(set(cache['domains']))

    methods = _parse_list(args.methods)
    pairs = _parse_pairs(args.pairs)
    seeds = [int(seed) for seed in _parse_list(args.seeds)]

    out_dir = Path(DATA_DIR) / 'pcgita_results'
    os.makedirs(out_dir, exist_ok=True)
    progress_path = out_dir / 'pcgita_spectrogram_progress.txt'
    partial_path = out_dir / 'pcgita_spectrogram_results_partial.csv'
    total_runs = len(pairs) * len(seeds) * len(methods)
    done_runs = 0

    def log_progress(message: str) -> None:
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
        print(line, flush=True)
        with open(progress_path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    progress_path.write_text(
        f'PC-GITA spektrogramovy beh: total_runs={total_runs}, '
        f'pairs={args.pairs}, methods={args.methods}, seeds={args.seeds}, '
        f'epochs={args.epochs}, batch_size={args.batch_size}\n',
        encoding='utf-8',
    )

    rows = []
    for source, target in pairs:
        sources = _source_domains(source, target, all_domains)
        split = _make_split(
            cache,
            sources,
            target,
            split_seed=args.split_seed,
            target_test_size=args.target_test_size,
        )
        print(
            f'\n{source} -> {target}: source_domains={"+".join(sources)}, '
            f'source_files={len(split.source_idx)}, '
            f'target_unlab_files={len(split.target_train_idx)}, '
            f'test_patients={len(split.test_patients)}',
            flush=True,
        )
        for seed in seeds:
            for method in methods:
                t0 = time.time()
                log_progress(
                    f'START {done_runs + 1}/{total_runs}: '
                    f'{source}->{target}, seed={seed}, method={method}'
                )
                metrics = _run_method(
                    cache,
                    split,
                    method,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                )
                row = {
                    'source': source,
                    'target': target,
                    'method': method,
                    'seed': seed,
                    'n_source_files': len(split.source_idx),
                    'n_target_train_files': len(split.target_train_idx),
                    'n_target_test_patients': len(split.test_patients),
                    **metrics,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                done_runs += 1
                log_progress(
                    f"DONE  {done_runs}/{total_runs}: "
                    f"{source}->{target}, seed={seed}, method={method}, "
                    f"AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, "
                    f"time={time.time() - t0:.1f}s"
                )

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / 'pcgita_spectrogram_results.csv', index=False)
    summary = summarize(results)
    summary.to_csv(out_dir / 'pcgita_spectrogram_summary.csv', index=False)
    print('\nSpectrogram CNN summary:')
    cols = ['source', 'target', 'method', 'auc_mean', 'auc_std', 'f1_mean',
            'sensitivity_mean', 'specificity_mean']
    print(summary[cols].to_string(index=False))
    print(f'\nOutputs saved in: {out_dir}')
    print(f'Total time: {time.time() - start:.1f}s')


if __name__ == '__main__':
    main()
