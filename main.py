"""Rychla kontrola hlavnej PC-GITA vetvy projektu.

Tento subor nespusta dlhe experimenty. Ma iba overit, ci su data/cache
na mieste, vypisat zakladne pocty a ukazat, kde su finalne vysledky.
Dlhy trening sa spusta cez samostatne skripty z README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DEVICE
from src.data.pcgita import (
    PCGITA_DIR,
    PCGITA_PATIENT_DOMAIN_FEATURES,
    build_pcgita_feature_cache,
    get_feature_columns,
)

DATA_RESULT_FILES = [
    'final_pairwise_best.csv',
    'final_multisource_best.csv',
    'pcgita_spectrogram_best.csv',
    'final_pairwise_best_auc.svg',
    'final_multisource_best_auc.svg',
    'pcgita_spectrogram_best_auc.svg',
]

REPORT_FILES = [
    '01_pairwise_best.csv',
    '01_pairwise_best_auc.png',
    '02_multisource_best.csv',
    '02_multisource_best_auc.png',
    '03_spectrogram_cnn_best.csv',
    '03_spectrogram_cnn_best_auc.png',
    '04_pairwise_all_methods_heatmap.png',
    '05_domain_gap_auc.png',
    '06_cnn_vs_acoustic_auc.png',
]


def _print_commands() -> None:
    print('\nOdporucane prikazy:')
    print(
        '  .\\.venv\\Scripts\\python.exe run_pcgita_experiments.py '
        '--methods logreg_source,svm_source,extra_trees,coral_linear,dann,mmd,deep_coral '
        '--seeds 42,7,123,2024,99 --epochs 20 --da-epochs 30 '
        '--pairs "words->sentences2,sentences2->words,vowels->ddk,ddk->vowels,'
        'sentences->read_text,read_text->sentences" --skip-multisource'
    )
    print('  powershell -ExecutionPolicy Bypass -File .\\run_pcgita_spectrogram_watch.ps1')


def _load_or_build_features(force: bool) -> pd.DataFrame | None:
    if PCGITA_PATIENT_DOMAIN_FEATURES.exists() and not force:
        return pd.read_csv(PCGITA_PATIENT_DOMAIN_FEATURES)
    if PCGITA_DIR.exists():
        return build_pcgita_feature_cache(force=force)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Kontrola PC-GITA projektu')
    parser.add_argument('--rebuild-features', action='store_true')
    parser.add_argument('--commands', action='store_true')
    args = parser.parse_args()

    print('PC-GITA domain adaptation projekt')
    print(f'Zariadenie pre torch: {DEVICE}')

    df = _load_or_build_features(force=args.rebuild_features)
    if df is None:
        print('\nPC-GITA data ani cache nie su lokalne dostupne.')
        print(f'Ocakavana cesta: {PCGITA_DIR}')
    else:
        feature_cols = get_feature_columns(df)
        print(f'\nPatient-domain tabulka: {df.shape[0]} riadkov, {len(feature_cols)} priznakov')
        print('\nPocty podla domeny a triedy:')
        print(df.groupby(['domain', 'label']).size().to_string())

    result_dir = Path(DATA_DIR) / 'pcgita_results'
    print('\nLokalne vystupy z data/pcgita_results:')
    for name in DATA_RESULT_FILES:
        path = result_dir / name
        status = 'OK' if path.exists() else 'chyba'
        print(f'  [{status}] {path}')

    report_dir = Path('reports') / 'pcgita'
    print('\nReport artefakty vhodne do repozitara:')
    for name in REPORT_FILES:
        path = report_dir / name
        status = 'OK' if path.exists() else 'chyba'
        print(f'  [{status}] {path}')

    if df is None:
        print('\nBez PC-GITA dat sa daju overit report tabulky a grafy.')
        print('Na kompletny prepocet experimentov treba doplnit PC-GITA do data/.')

    if args.commands:
        _print_commands()


if __name__ == '__main__':
    main()
