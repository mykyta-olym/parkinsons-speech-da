"""PC-GITA experimenty nad embeddings z predtrenovaneho recoveho modelu.

Druha vetva pipeline: namiesto rucnych akustickych priznakov pouzijeme self-
supervised reprezentacie z wav2vec~2.0 / HuBERT a spustime presne ten isty
protokol (in-domain CV, pairwise cross-task, multi-source) ako klasicka
tabular vetva. Vsetky DA metody (Subspace, Linear CORAL, Deep CORAL, MMD,
DANN, CDAN) sa pouzivaju nezmenene - jediny rozdiel je vstupna feature
matica.

Postup:
    1. ``src/data/pcgita_pretrained.py`` extrahuje frozen wav2vec~2 embeddings,
       sprava mean+std time pooling a vystupy agreguje na (pacient, doména) -
       presne tak, ako tabular vetva v ``src/data/pcgita.py``.
    2. Tento skript nacita patient-domain CSV (1536-dim default pre wav2vec2-base)
       a zavolá ``run_in_domain_cv`` / ``run_cross_domain`` / ``run_multisource``
       z hlavneho experimentalneho skriptu, takze sa nedupliuje ziadna logika.
    3. Vystupy sa ulozia do ``data/pcgita_pretrained_results/`` so suffixom
       odvodenym od mena modelu (``facebook__wav2vec2-base`` atd.).

Spustenie::

    .\\.venv\\Scripts\\python.exe run_pcgita_pretrained_experiments.py \\
        --model facebook/wav2vec2-base --segment-seconds 4 \\
        --batch-size 8 --epochs 30 --da-epochs 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from run_pcgita_experiments import (
    DEFAULT_DOMAINS,
    DEFAULT_METHODS,
    DEFAULT_PAIRS,
    _parse_list,
    _parse_pairs,
    run_cross_domain,
    run_in_domain_cv,
    run_multisource,
    summarize,
)
from src.config import DATA_DIR, DEVICE, SEEDS
from src.data.pcgita import get_feature_columns
from src.data.pcgita_pretrained import (
    DEFAULT_MODEL,
    DEFAULT_SEGMENT_SECONDS,
    build_pretrained_patient_domain,
    pretrained_cache_path,
)


def _output_dir(model_name: str) -> Path:
    slug = model_name.replace('/', '__').replace(' ', '_')
    out_dir = Path(DATA_DIR) / 'pcgita_pretrained_results' / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description='PC-GITA DA experimenty nad embeddings z predtrenovaneho rec. modelu.'
    )
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Huggingface model id (wav2vec2 / HuBERT). '
                             f'Default: {DEFAULT_MODEL}')
    parser.add_argument('--segment-seconds', type=float, default=DEFAULT_SEGMENT_SECONDS,
                        help='Dlzka chunku pre encoder. Default 4 s.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch chunkov pri extrakcii embeddings.')
    parser.add_argument('--force-features', action='store_true',
                        help='Prepocita embeddings aj ked existuje cache.')
    parser.add_argument('--cache-only', action='store_true',
                        help='Iba extrahuj a uloz embeddings, neziel experiment.')
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS),
                        help='Comma-separated DA metody.')
    parser.add_argument('--pairs', default=','.join(f'{s}->{t}' for s, t in DEFAULT_PAIRS),
                        help='Comma-separated source->target dvojice.')
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in SEEDS[:3]),
                        help='Comma-separated random seeds.')
    parser.add_argument('--clf-batch-size', type=int, default=16,
                        help='Batch size pre tieniny DA klasifikatorov (torch).')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--da-epochs', type=int, default=50)
    parser.add_argument('--target-test-size', type=float, default=0.4)
    parser.add_argument('--split-seed', type=int, default=12345)
    parser.add_argument('--skip-multisource', action='store_true')
    parser.add_argument('--skip-in-domain', action='store_true')
    parser.add_argument('--skip-pairwise', action='store_true')
    args = parser.parse_args()

    start = time.time()
    print(f'Device: {DEVICE}')
    cache_path = pretrained_cache_path(args.model)
    print(f'Pretrained patient-domain cache: {cache_path}')

    df = build_pretrained_patient_domain(
        model_name=args.model,
        segment_seconds=args.segment_seconds,
        batch_size=args.batch_size,
        force=args.force_features,
        cache_path=cache_path,
    )
    feature_cols = get_feature_columns(df)
    print(f'PC-GITA pretrained table: {df.shape}, embedding_dim={len(feature_cols)}')
    print(df.groupby(['domain', 'label']).size().to_string())

    if args.cache_only:
        print('\n[cache-only] hotovo, experiment vynechany.')
        return

    methods = _parse_list(args.methods)
    pairs = _parse_pairs(args.pairs)
    seeds = [int(seed) for seed in _parse_list(args.seeds)]
    out_dir = _output_dir(args.model)

    if not args.skip_in_domain:
        in_domain = run_in_domain_cv(df, DEFAULT_DOMAINS, seed=args.split_seed)
        in_domain.to_csv(out_dir / 'pretrained_in_domain_cv.csv', index=False)
        print('\n[pretrained] In-domain CV (LogReg nad embeddings):')
        print(
            in_domain.groupby('domain')[['auc', 'f1', 'balanced_accuracy']]
            .agg(['mean', 'std']).round(3).to_string()
        )

    all_results = []
    if not args.skip_pairwise:
        pair_results = run_cross_domain(
            df,
            pairs=pairs,
            methods=methods,
            seeds=seeds,
            epochs=args.epochs,
            da_epochs=args.da_epochs,
            batch_size=args.clf_batch_size,
            target_test_size=args.target_test_size,
            split_seed=args.split_seed,
        )
        pair_results.to_csv(out_dir / 'pretrained_pair_results.csv', index=False)
        all_results.append(pair_results)

    if not args.skip_multisource:
        multi_methods = [m for m in methods if m not in {'subspace'}]
        targets = sorted({target for _, target in pairs}) if pairs else DEFAULT_DOMAINS
        multi_results = run_multisource(
            df,
            targets=targets,
            methods=multi_methods,
            seeds=seeds,
            epochs=args.epochs,
            da_epochs=args.da_epochs,
            batch_size=args.clf_batch_size,
            target_test_size=args.target_test_size,
            split_seed=args.split_seed,
        )
        multi_results.to_csv(out_dir / 'pretrained_multisource_results.csv', index=False)
        all_results.append(multi_results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        summary = summarize(combined)
        summary.to_csv(out_dir / 'pretrained_summary.csv', index=False)
        print('\n[pretrained] Cross-domain summary (head):')
        cols = ['source', 'target', 'method', 'auc_mean', 'auc_std', 'f1_mean',
                'sensitivity_mean', 'specificity_mean']
        print(summary[cols].head(40).to_string(index=False))

    print(f'\n[pretrained] Outputs in: {out_dir}')
    print(f'Total time: {time.time() - start:.1f}s')


if __name__ == '__main__':
    main()
