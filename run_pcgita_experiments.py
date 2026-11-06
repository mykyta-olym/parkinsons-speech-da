"""PC-GITA experimenty nad audio priznakmi.

Domeny su recove ulohy v PC-GITA. Jedna vzorka je jeden pacient v jednej
domene. Split je patient-wise, cize pacient z testu nie je ani v source
treningu, ani v target-unlabeled casti. Target labely sa pouziju iba pri
vypocte metrik.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from src.config import DATA_DIR, DEVICE, SEEDS
from src.data.datasets import ParkinsonDataset, compute_class_weights
from src.data.pcgita import build_pcgita_feature_cache, get_feature_columns
from src.models import MLP, CDANModel, CORALModel, DANNModel, MMDModel, SubspaceAlignmentDA
from src.training import train_cdan, train_coral, train_dann, train_mmd, train_model
from src.utils import set_seed

DEFAULT_DOMAINS = [
    'vowels',
    'ddk',
    'words',
    'sentences',
    'sentences2',
    'read_text',
    'monologue',
]

DEFAULT_PAIRS = [
    ('vowels', 'ddk'),
    ('ddk', 'vowels'),
    ('words', 'sentences2'),
    ('sentences2', 'words'),
    ('sentences', 'read_text'),
    ('read_text', 'sentences'),
]

DEFAULT_METHODS = [
    'mlp_source',
    'logreg_source',
    'svm_source',
    'extra_trees',
    'subspace',
    'coral_linear',
    'dann',
    'mmd',
    'deep_coral',
    'cdan',
]


@dataclass
class SplitData:
    X_src: np.ndarray
    y_src: np.ndarray
    X_tgt_train: np.ndarray
    y_tgt_train: np.ndarray
    X_tgt_test: np.ndarray
    y_tgt_test: np.ndarray
    train_patients: list[str]
    test_patients: list[str]


def _parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(',') if part.strip()]


def _parse_pairs(value: str) -> list[tuple[str, str]]:
    pairs = []
    for item in _parse_list(value):
        if '->' not in item:
            raise ValueError(f'Par musi mat format source->target: {item}')
        source, target = item.split('->', 1)
        pairs.append((source.strip(), target.strip()))
    return pairs


def _make_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    source_domains: list[str],
    target_domain: str,
    split_seed: int,
    target_test_size: float,
) -> SplitData:
    target_df = df[df['domain'] == target_domain].copy()
    if target_df.empty:
        raise ValueError(f'Cielova domena neexistuje: {target_domain}')

    groups = target_df['patient_id'].to_numpy()
    y_target = target_df['label'].to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=target_test_size,
        random_state=split_seed,
    )
    train_idx, test_idx = next(splitter.split(target_df, y_target, groups=groups))

    train_patients = sorted(set(target_df.iloc[train_idx]['patient_id']))
    test_patients = sorted(set(target_df.iloc[test_idx]['patient_id']))

    source_df = df[
        df['domain'].isin(source_domains)
        & df['patient_id'].isin(train_patients)
    ].copy()
    target_train_df = target_df.iloc[train_idx].copy()
    target_test_df = target_df.iloc[test_idx].copy()

    return SplitData(
        X_src=source_df[feature_cols].to_numpy(dtype=np.float32),
        y_src=source_df['label'].to_numpy(dtype=np.int64),
        X_tgt_train=target_train_df[feature_cols].to_numpy(dtype=np.float32),
        y_tgt_train=target_train_df['label'].to_numpy(dtype=np.int64),
        X_tgt_test=target_test_df[feature_cols].to_numpy(dtype=np.float32),
        y_tgt_test=target_test_df['label'].to_numpy(dtype=np.int64),
        train_patients=train_patients,
        test_patients=test_patients,
    )


def _standardize(split: SplitData):
    source_scaler = StandardScaler().fit(split.X_src)
    target_scaler = StandardScaler().fit(split.X_tgt_train)
    return (
        source_scaler.transform(split.X_src).astype(np.float32),
        target_scaler.transform(split.X_tgt_train).astype(np.float32),
        target_scaler.transform(split.X_tgt_test).astype(np.float32),
    )


def _loaders(split: SplitData, seed: int, batch_size: int):
    Xs, Xt_train, Xt_test = _standardize(split)
    generator = torch.Generator().manual_seed(seed)
    src_loader = DataLoader(
        ParkinsonDataset(Xs, split.y_src),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    tgt_loader = DataLoader(
        ParkinsonDataset(Xt_train, split.y_tgt_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        ParkinsonDataset(Xt_test, split.y_tgt_test),
        batch_size=batch_size * 4,
        shuffle=False,
    )
    return src_loader, tgt_loader, test_loader


def _torch_scores(model, test_loader, model_type: str) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(DEVICE)
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            if model_type in {'dann', 'cdan'}:
                logits, _, _ = model(features)
            elif model_type in {'mmd', 'deep_coral'}:
                logits, _ = model(features)
            else:
                logits = model(features)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_true.append(labels.numpy())
            y_score.append(probs)
    return np.concatenate(y_true), np.concatenate(y_score)


def _matrix_sqrt(cov: np.ndarray, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 1e-6, None)
    if inverse:
        values = 1.0 / np.sqrt(values)
    else:
        values = np.sqrt(values)
    return (vectors * values) @ vectors.T


def _coral_linear_scores(split: SplitData, seed: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, Xt_train, Xt_test = _standardize(split)
    reg = 1e-3 * np.eye(Xs.shape[1])
    cov_s = np.cov(Xs, rowvar=False) + reg
    cov_t = np.cov(Xt_train, rowvar=False) + reg
    transform = _matrix_sqrt(cov_s, inverse=True) @ _matrix_sqrt(cov_t, inverse=False)
    Xs_aligned = Xs @ transform

    clf = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=seed,
    )
    clf.fit(Xs_aligned, split.y_src)
    return split.y_tgt_test, clf.predict_proba(Xt_test)[:, 1]


def _sklearn_scores(method: str, split: SplitData, seed: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, _, Xt_test = _standardize(split)
    if method == 'logreg_source':
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)
        clf.fit(Xs, split.y_src)
        return split.y_tgt_test, clf.predict_proba(Xt_test)[:, 1]
    if method == 'svm_source':
        clf = SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=seed,
        )
        clf.fit(Xs, split.y_src)
        return split.y_tgt_test, clf.predict_proba(Xt_test)[:, 1]
    if method == 'extra_trees':
        clf = ExtraTreesClassifier(
            n_estimators=400,
            class_weight='balanced',
            random_state=seed,
        )
        clf.fit(Xs, split.y_src)
        return split.y_tgt_test, clf.predict_proba(Xt_test)[:, 1]
    if method == 'subspace':
        n_comp = min(16, split.X_src.shape[1], split.X_src.shape[0] - 1,
                     split.X_tgt_train.shape[0] - 1)
        model = SubspaceAlignmentDA(n_components=n_comp, random_state=seed)
        model.fit(split.X_src, split.y_src, split.X_tgt_train)
        return split.y_tgt_test, model.predict_proba(split.X_tgt_test)[:, 1]
    if method == 'coral_linear':
        return _coral_linear_scores(split, seed)
    raise ValueError(f'Neznáma sklearn metóda: {method}')


def _run_method(
    method: str,
    split: SplitData,
    seed: int,
    input_size: int,
    epochs: int,
    da_epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    if method in {'logreg_source', 'svm_source', 'extra_trees', 'subspace', 'coral_linear'}:
        return _sklearn_scores(method, split, seed)

    src_loader, tgt_loader, test_loader = _loaders(split, seed, batch_size)
    class_weights = compute_class_weights(split.y_src)

    if method == 'mlp_source':
        model = MLP(input_size=input_size)
        train_model(model, src_loader, num_epochs=epochs, class_weights=class_weights)
        return _torch_scores(model, test_loader, 'standard')
    if method == 'dann':
        model = DANNModel(input_size=input_size)
        train_dann(model, src_loader, tgt_loader, num_epochs=da_epochs,
                   class_weights=class_weights)
        return _torch_scores(model, test_loader, 'dann')
    if method == 'mmd':
        model = MMDModel(input_size=input_size)
        train_mmd(model, src_loader, tgt_loader, num_epochs=da_epochs,
                  class_weights=class_weights)
        return _torch_scores(model, test_loader, 'mmd')
    if method == 'deep_coral':
        model = CORALModel(input_size=input_size)
        train_coral(model, src_loader, tgt_loader, num_epochs=da_epochs,
                    class_weights=class_weights)
        return _torch_scores(model, test_loader, 'deep_coral')
    if method == 'cdan':
        model = CDANModel(input_size=input_size)
        train_cdan(model, src_loader, tgt_loader, num_epochs=da_epochs,
                   class_weights=class_weights)
        return _torch_scores(model, test_loader, 'cdan')
    raise ValueError(f'Neznáma metóda: {method}')


def _metric_row(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str,
    source: str,
    target: str,
    seed: int,
    n_source: int,
    n_target_train: int,
    n_target_test: int,
) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'source': source,
        'target': target,
        'method': method,
        'seed': seed,
        'auc': roc_auc_score(y_true, y_score),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'n_source': n_source,
        'n_target_train': n_target_train,
        'n_target_test': n_target_test,
        'target_pd_test': int(np.sum(y_true)),
        'target_hc_test': int(len(y_true) - np.sum(y_true)),
    }


def run_cross_domain(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    methods: list[str],
    seeds: list[int],
    epochs: int,
    da_epochs: int,
    batch_size: int,
    target_test_size: float,
    split_seed: int,
) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    rows = []
    for source, target in pairs:
        split = _make_split(
            df,
            feature_cols,
            [source],
            target,
            split_seed=split_seed,
            target_test_size=target_test_size,
        )
        print(
            f'\n{source} -> {target}: source={len(split.y_src)}, '
            f'target_unlab={len(split.y_tgt_train)}, test={len(split.y_tgt_test)}'
        )
        for seed in seeds:
            for method in methods:
                start = time.time()
                y_true, y_score = _run_method(
                    method,
                    split,
                    seed,
                    input_size=len(feature_cols),
                    epochs=epochs,
                    da_epochs=da_epochs,
                    batch_size=batch_size,
                )
                row = _metric_row(
                    y_true,
                    y_score,
                    method,
                    source,
                    target,
                    seed,
                    len(split.y_src),
                    len(split.y_tgt_train),
                    len(split.y_tgt_test),
                )
                rows.append(row)
                print(
                    f"  seed={seed:>4} {method:<13} "
                    f"AUC={row['auc']:.3f} F1={row['f1']:.3f} "
                    f"({time.time() - start:.1f}s)"
                )
    return pd.DataFrame(rows)


def run_multisource(
    df: pd.DataFrame,
    targets: list[str],
    methods: list[str],
    seeds: list[int],
    epochs: int,
    da_epochs: int,
    batch_size: int,
    target_test_size: float,
    split_seed: int,
) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    all_domains = sorted(df['domain'].unique())
    rows = []
    for target in targets:
        sources = [domain for domain in all_domains if domain != target]
        split = _make_split(
            df,
            feature_cols,
            sources,
            target,
            split_seed=split_seed,
            target_test_size=target_test_size,
        )
        source_name = '+'.join(sources)
        print(
            f'\nmultisource -> {target}: source={len(split.y_src)}, '
            f'target_unlab={len(split.y_tgt_train)}, test={len(split.y_tgt_test)}'
        )
        for seed in seeds:
            for method in methods:
                start = time.time()
                y_true, y_score = _run_method(
                    method,
                    split,
                    seed,
                    input_size=len(feature_cols),
                    epochs=epochs,
                    da_epochs=da_epochs,
                    batch_size=batch_size,
                )
                row = _metric_row(
                    y_true,
                    y_score,
                    method,
                    source_name,
                    target,
                    seed,
                    len(split.y_src),
                    len(split.y_tgt_train),
                    len(split.y_tgt_test),
                )
                row['source_mode'] = 'multi'
                rows.append(row)
                print(
                    f"  seed={seed:>4} {method:<13} "
                    f"AUC={row['auc']:.3f} F1={row['f1']:.3f} "
                    f"({time.time() - start:.1f}s)"
                )
    return pd.DataFrame(rows)


def run_in_domain_cv(df: pd.DataFrame, domains: list[str], seed: int) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    rows = []
    for domain in domains:
        subset = df[df['domain'] == domain].copy()
        X = subset[feature_cols].to_numpy(dtype=np.float32)
        y = subset['label'].to_numpy(dtype=np.int64)
        groups = subset['patient_id'].to_numpy()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            scaler = StandardScaler().fit(X[train_idx])
            clf = LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                random_state=seed,
            )
            clf.fit(scaler.transform(X[train_idx]), y[train_idx])
            y_score = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
            y_pred = (y_score >= 0.5).astype(int)
            rows.append({
                'domain': domain,
                'fold': fold,
                'method': 'logreg_in_domain',
                'auc': roc_auc_score(y[test_idx], y_score),
                'f1': f1_score(y[test_idx], y_pred, zero_division=0),
                'balanced_accuracy': balanced_accuracy_score(y[test_idx], y_pred),
            })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ['auc', 'f1', 'balanced_accuracy', 'sensitivity', 'specificity']
    group_cols = ['source', 'target', 'method']
    summary = (
        results
        .groupby(group_cols, dropna=False)[metric_cols]
        .agg(['mean', 'std', 'min', 'max'])
        .reset_index()
    )
    summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary.sort_values(['target', 'auc_mean'], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser(description='PC-GITA domain adaptation experimenty')
    parser.add_argument('--force-features', action='store_true',
                        help='Recompute audio features from scratch.')
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS),
                        help='Comma-separated method list.')
    parser.add_argument('--pairs', default=','.join(f'{s}->{t}' for s, t in DEFAULT_PAIRS),
                        help='Comma-separated source->target pairs.')
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in SEEDS[:3]),
                        help='Comma-separated random seeds.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--da-epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--target-test-size', type=float, default=0.4)
    parser.add_argument('--split-seed', type=int, default=12345)
    parser.add_argument('--skip-multisource', action='store_true')
    args = parser.parse_args()

    start = time.time()
    print(f'Device: {DEVICE}')
    df = build_pcgita_feature_cache(force=args.force_features)
    feature_cols = get_feature_columns(df)
    print(f'PC-GITA patient-domain table: {df.shape}, features={len(feature_cols)}')
    print(df.groupby(['domain', 'label']).size().to_string())

    methods = _parse_list(args.methods)
    pairs = _parse_pairs(args.pairs)
    seeds = [int(seed) for seed in _parse_list(args.seeds)]

    out_dir = Path(DATA_DIR) / 'pcgita_results'
    os.makedirs(out_dir, exist_ok=True)

    in_domain = run_in_domain_cv(df, DEFAULT_DOMAINS, seed=args.split_seed)
    in_domain.to_csv(out_dir / 'pcgita_in_domain_cv.csv', index=False)

    pair_results = run_cross_domain(
        df,
        pairs=pairs,
        methods=methods,
        seeds=seeds,
        epochs=args.epochs,
        da_epochs=args.da_epochs,
        batch_size=args.batch_size,
        target_test_size=args.target_test_size,
        split_seed=args.split_seed,
    )
    pair_results.to_csv(out_dir / 'pcgita_pair_results.csv', index=False)

    all_results = [pair_results]
    if not args.skip_multisource:
        multi_methods = [m for m in methods if m not in {'subspace'}]
        targets = sorted({target for _, target in pairs})
        multi_results = run_multisource(
            df,
            targets=targets,
            methods=multi_methods,
            seeds=seeds,
            epochs=args.epochs,
            da_epochs=args.da_epochs,
            batch_size=args.batch_size,
            target_test_size=args.target_test_size,
            split_seed=args.split_seed,
        )
        multi_results.to_csv(out_dir / 'pcgita_multisource_results.csv', index=False)
        all_results.append(multi_results)

    combined = pd.concat(all_results, ignore_index=True)
    summary = summarize(combined)
    summary.to_csv(out_dir / 'pcgita_summary.csv', index=False)

    print('\nIn-domain CV summary:')
    print(in_domain.groupby('domain')[['auc', 'f1', 'balanced_accuracy']].agg(['mean', 'std']))
    print('\nCross-domain summary:')
    cols = ['source', 'target', 'method', 'auc_mean', 'auc_std', 'f1_mean',
            'sensitivity_mean', 'specificity_mean']
    print(summary[cols].to_string(index=False))
    print(f'\nOutputs saved in: {out_dir}')
    print(f'Total time: {time.time() - start:.1f}s')


if __name__ == '__main__':
    main()
