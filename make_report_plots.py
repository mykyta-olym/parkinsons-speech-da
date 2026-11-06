"""Publikacne grafy z finalnych CSV tabuliek v reports/pcgita/.

Skript nepouziva raw PC-GITA data. Staci priecinok reports/pcgita s CSV
sumarmi a da sa spustit aj po cistom klone repozitara. Vystupom su PNG (220
DPI) a PDF (vektorovo do LaTeX dokumentu) v rovnakom priecinku.

Pouzite:
    python make_report_plots.py
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPORT_DIR = Path('reports') / 'pcgita'
PRETRAINED_DIR = Path('data') / 'pcgita_pretrained_results' / 'facebook__wav2vec2-base'

# ColorBrewer-friendly paleta podla typu metody.
# Hodnoty su konzistentne s Tab10 / "muted" stylom z paperov.
COLOR_BASELINE = '#4C72B0'      # modra: logreg, svm, extra_trees, mlp_source
COLOR_CLASSICAL = '#DD8452'     # oranzova: coral_linear, subspace
COLOR_DEEP = '#55A868'          # zelena: mmd, deep_coral
COLOR_ADVERSARIAL = '#8172B3'   # fialova: dann, cdan
COLOR_CONTRASTIVE = '#937860'   # hneda: contrastive
COLOR_CNN = '#64B5CD'           # tyrkysova: cnn_*
COLOR_WINNER = '#C5A059'        # zlta: zvyraznenie najlepsieho
COLOR_GRID = '#D0D3D7'
COLOR_TEXT_DIM = '#5A6772'

BASELINE = {'logreg_source', 'svm_source', 'extra_trees', 'mlp_source'}
CLASSICAL = {'coral_linear', 'subspace'}
DEEP = {'mmd', 'deep_coral'}
ADVERSARIAL = {'dann', 'cdan'}
CONTRASTIVE = {'contrastive'}


def _color_for(method: str) -> str:
    method = (method or '').lower()
    if method in BASELINE:
        return COLOR_BASELINE
    if method in CLASSICAL:
        return COLOR_CLASSICAL
    if method in DEEP:
        return COLOR_DEEP
    if method in ADVERSARIAL:
        return COLOR_ADVERSARIAL
    if method in CONTRASTIVE:
        return COLOR_CONTRASTIVE
    if method.startswith('cnn'):
        return COLOR_CNN
    return COLOR_BASELINE


def _category_for(method: str) -> str:
    method = (method or '').lower()
    if method in BASELINE:
        return 'baseline'
    if method in CLASSICAL:
        return 'classical-DA'
    if method in DEEP:
        return 'deep-DA'
    if method in ADVERSARIAL:
        return 'adversarial-DA'
    if method in CONTRASTIVE:
        return 'contrastive-DA'
    if method.startswith('cnn'):
        return 'CNN'
    return 'baseline'


def _legend_handles(categories: Iterable[str]) -> list[Line2D]:
    seen = []
    handles = []
    mapping = {
        'baseline': ('Baseline (source-only)', COLOR_BASELINE),
        'classical-DA': ('Klasicka DA (CORAL, Subspace)', COLOR_CLASSICAL),
        'deep-DA': ('Hlboka DA (MMD, Deep CORAL)', COLOR_DEEP),
        'adversarial-DA': ('Adversariálna DA (DANN, CDAN)', COLOR_ADVERSARIAL),
        'contrastive-DA': ('Contrastive DA', COLOR_CONTRASTIVE),
        'CNN': ('CNN (log-mel)', COLOR_CNN),
    }
    for cat in categories:
        if cat in seen or cat not in mapping:
            continue
        seen.append(cat)
        label, color = mapping[cat]
        handles.append(
            Line2D([0], [0], marker='s', linestyle='', markersize=10,
                   markerfacecolor=color, markeredgecolor=color, label=label)
        )
    return handles


def _style() -> None:
    mpl.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 220,
        'figure.constrained_layout.use': False,
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'normal',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#2A2F36',
        'axes.linewidth': 0.9,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': COLOR_GRID,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _header(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Place a 2-line header above the axes; rezerve top space cez tight_layout."""
    fig.text(0.02, 0.965, title, ha='left', va='top',
             fontsize=14, fontweight='bold', color='#17202A')
    fig.text(0.02, 0.918, subtitle, ha='left', va='top',
             fontsize=10, color=COLOR_TEXT_DIM)


def _save(fig: plt.Figure, name: str, top: float = 0.88) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    base = REPORT_DIR / name
    fig.savefig(base.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# G1, G2, G3 — najlepsi model pre par/target (horizontal bar)
# ----------------------------------------------------------------------------

def _barh_best(
    df: pd.DataFrame,
    out_name: str,
    title: str,
    subtitle: str,
    label_col: str,
    annotate_method: bool = True,
) -> None:
    df = df.sort_values('auc_mean', ascending=True).reset_index(drop=True)
    if label_col == 'pair':
        labels = df['source'].astype(str) + r' $\rightarrow$ ' + df['target'].astype(str)
    else:
        labels = df[label_col].astype(str)

    methods = df['method'].astype(str)
    colors = [_color_for(m) for m in methods]
    aucs = df['auc_mean'].astype(float).to_numpy()
    stds = df.get('auc_std', pd.Series(np.zeros(len(df)))).astype(float).to_numpy()

    n = len(df)
    fig, ax = plt.subplots(figsize=(10.6, max(3.8, 0.62 * n + 2.4)))
    y = np.arange(n)
    ax.barh(y, aucs, xerr=stds, color=colors, alpha=0.92,
            edgecolor='white', linewidth=0.6,
            error_kw=dict(ecolor='#2A2F36', elinewidth=1.0, capsize=3))
    ax.set_yticks(y, labels)
    xmax = float(min(0.99, max(0.92, (aucs + stds).max() + 0.10)))
    ax.set_xlim(0.50, xmax)
    ax.set_xlabel('AUC')

    _header(fig, title, subtitle)

    ax.axvline(0.5, color='#9CA3AF', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.text(0.502, n - 0.5, 'chance', color='#9CA3AF', fontsize=9,
            ha='left', va='top')

    # Anotacie napravo od error bar
    for i, (auc, std, method) in enumerate(zip(aucs, stds, methods, strict=True)):
        text_x = auc + std + 0.012
        if annotate_method:
            text = f'{auc:.3f}  ({method})'
        else:
            text = f'{auc:.3f}'
        ax.text(text_x, i, text, va='center', fontsize=10, color='#1F2937')

    handles = _legend_handles([_category_for(m) for m in methods])
    if handles:
        ax.legend(handles=handles, loc='lower right', borderaxespad=0.5)
    _save(fig, out_name, top=0.88)


def plot_pairwise_best() -> None:
    df = pd.read_csv(REPORT_DIR / '01_pairwise_best.csv')
    if 'pair' not in df.columns:
        df['pair'] = df['source'] + ' -> ' + df['target']
    df.to_csv(REPORT_DIR / '01_pairwise_best.csv', index=False)
    _barh_best(
        df,
        '01_pairwise_best_auc',
        'Pairwise cross-domain: najlepsi model pre kazdy par',
        'AUC priemer +/- std cez 5 seedov; patient-wise split.',
        'pair',
    )


def plot_multisource_best() -> None:
    df = pd.read_csv(REPORT_DIR / '02_multisource_best.csv')
    _barh_best(
        df,
        '02_multisource_best_auc',
        'Multi-source: najlepsi model pre kazdu cielovu domenu',
        'Source = vsetky ostatne PC-GITA ulohy; 5 seedov.',
        'target',
    )


def plot_spectrogram_best() -> None:
    df = pd.read_csv(REPORT_DIR / '03_spectrogram_cnn_best.csv')
    _barh_best(
        df,
        '03_spectrogram_cnn_best_auc',
        'Log-mel CNN: najlepsi model pre kazdu cielovu domenu',
        'Patient-level evaluacia (priemer score po suboroch pacienta); 2 seedy.',
        'target',
    )


# ----------------------------------------------------------------------------
# G4 — heatmapa AUC pre vsetky metody x pary
# ----------------------------------------------------------------------------

def plot_pairwise_heatmap() -> None:
    df = pd.read_csv(REPORT_DIR / '04_pairwise_all_methods.csv').copy()
    df['pair'] = df['source'].astype(str) + r' $\rightarrow$ ' + df['target'].astype(str)
    pivot = df.pivot(index='pair', columns='method', values='auc_mean')

    method_order = [
        c for c in [
            'logreg_source', 'svm_source', 'extra_trees',
            'coral_linear', 'mmd', 'deep_coral', 'dann',
        ] if c in pivot.columns
    ]
    pivot = pivot[method_order]
    # Zoradenie riadkov podla maxima cez metody (najlahsie pary hore)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(11.5, 0.7 * len(pivot) + 2.8))
    cmap = mpl.colormaps['YlGnBu'].resampled(256)
    image = ax.imshow(pivot.values, cmap=cmap, vmin=0.50, vmax=0.90, aspect='auto')

    _header(fig,
            'Pairwise cross-domain: AUC pre vsetky metody',
            'Tucne hodnoty: najlepsia metoda pre dany par. Zlty ramcek: najlepsia metoda v stlpci.')

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.tick_params(axis='both', length=0)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.4)
    ax.tick_params(which='minor', length=0)

    row_max_idx = pivot.values.argmax(axis=1)
    col_max_idx = pivot.values.argmax(axis=0)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isnan(value):
                continue
            is_row_best = (j == row_max_idx[i])
            text_color = '#0E2A47' if value < 0.78 else '#FFFFFF'
            weight = 'bold' if is_row_best else 'normal'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                    fontsize=10, color=text_color, fontweight=weight)

    # Ramcek pre najlepsiu metodu v stlpci
    for j, i in enumerate(col_max_idx):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='#C5A059',
                                    linewidth=2.0))

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('AUC priemer (5 seedov)')
    cbar.outline.set_visible(False)
    _save(fig, '04_pairwise_all_methods_heatmap', top=0.86)


# ----------------------------------------------------------------------------
# G5 — domain gap (in-domain vs OOD)
# ----------------------------------------------------------------------------

def plot_domain_gap() -> None:
    df = pd.read_csv(REPORT_DIR / '05_domain_gap.csv').copy()
    df = df.sort_values('drop_auc', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.4, 0.75 * len(df) + 2.4))
    y = np.arange(len(df))
    height = 0.36
    ax.barh(y - height / 2, df['in_domain_auc'], height,
            label='In-domain CV', color=COLOR_BASELINE,
            edgecolor='white', linewidth=0.6)
    ax.barh(y + height / 2, df['out_domain_auc'], height,
            label='Out-of-domain (multi-source)', color=COLOR_DEEP,
            edgecolor='white', linewidth=0.6)
    ax.set_yticks(y, df['target'])
    ax.set_xlim(0.55, 1.0)
    ax.set_xlabel('AUC')
    _header(fig,
            'Domain gap: in-domain CV vs out-of-domain prenos',
            'Modra = 5-fold patient-wise CV v doméne; zelena = najlepsi multi-source model na cielovej domene.')
    ax.legend(loc='lower right', ncol=1)

    for i, row in df.iterrows():
        x_text = max(row['in_domain_auc'], row['out_domain_auc']) + 0.012
        ax.text(x_text, i,
                f"pokles {row['drop_auc']:.3f}",
                va='center', fontsize=10, color='#B5413B' if row['drop_auc'] > 0.05 else '#1F2937')
        ax.text(row['in_domain_auc'] - 0.005, i - height / 2,
                f"{row['in_domain_auc']:.3f}",
                va='center', ha='right', fontsize=9, color='white')
        ax.text(row['out_domain_auc'] - 0.005, i + height / 2,
                f"{row['out_domain_auc']:.3f}",
                va='center', ha='right', fontsize=9, color='white')
    _save(fig, '05_domain_gap_auc', top=0.88)


# ----------------------------------------------------------------------------
# G6 — CNN vs akusticke priznaky
# ----------------------------------------------------------------------------

def plot_cnn_vs_acoustic() -> None:
    df = pd.read_csv(REPORT_DIR / '06_cnn_vs_acoustic.csv').copy()
    df = df.sort_values('acoustic_auc', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    x = np.arange(len(df))
    width = 0.36
    ax.bar(x - width / 2, df['acoustic_auc'], width,
           label='Akusticke priznaky (47-dim)', color=COLOR_BASELINE,
           edgecolor='white', linewidth=0.6)
    ax.bar(x + width / 2, df['cnn_auc'], width,
           label='Log-mel CNN', color=COLOR_CNN,
           edgecolor='white', linewidth=0.6)
    ax.set_xticks(x, df['target'])
    ax.set_ylim(0.50, 1.05)
    ax.set_ylabel('AUC')
    _header(fig,
            'CNN nad spektrogramami vs akusticke priznaky',
            'Cielova domena na osi x; obe metody trenovane v multi-source rezime.')
    ax.legend(loc='upper right')
    ax.axhline(0.5, color='#9CA3AF', linewidth=0.8, linestyle='--', alpha=0.7)

    for i, row in df.iterrows():
        ax.text(i - width / 2, row['acoustic_auc'] + 0.012,
                f"{row['acoustic_auc']:.3f}", ha='center', fontsize=9,
                color='#1F2937')
        ax.text(i + width / 2, row['cnn_auc'] + 0.012,
                f"{row['cnn_auc']:.3f}", ha='center', fontsize=9,
                color='#1F2937')
        diff = float(row['diff_cnn_minus_acoustic'])
        ax.text(i, max(row['acoustic_auc'], row['cnn_auc']) + 0.06,
                f"$\\Delta$={diff:+.3f}", ha='center', fontsize=9,
                color=(COLOR_DEEP if diff > 0 else '#B5413B'))
    _save(fig, '06_cnn_vs_acoustic_auc', top=0.88)


# ----------------------------------------------------------------------------
# G7 — distribucia AUC pre kazdu metodu (box-plot cez vsetky pary)
# ----------------------------------------------------------------------------

def plot_method_distribution() -> None:
    df = pd.read_csv(REPORT_DIR / '04_pairwise_all_methods.csv').copy()
    method_order = [
        m for m in [
            'extra_trees', 'logreg_source', 'svm_source',
            'coral_linear', 'mmd', 'deep_coral', 'dann',
        ] if m in df['method'].unique()
    ]
    data = [df[df['method'] == m]['auc_mean'].values for m in method_order]
    colors = [_color_for(m) for m in method_order]

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    parts = ax.boxplot(data, vert=True, patch_artist=True,
                       widths=0.55, showmeans=True,
                       medianprops=dict(color='#2A2F36', linewidth=1.4),
                       meanprops=dict(marker='D', markerfacecolor='white',
                                      markeredgecolor='#2A2F36', markersize=6),
                       whiskerprops=dict(color='#2A2F36', linewidth=1.0),
                       capprops=dict(color='#2A2F36', linewidth=1.0),
                       flierprops=dict(marker='o', markerfacecolor='#9CA3AF',
                                       markeredgecolor='#9CA3AF', markersize=5,
                                       alpha=0.7))
    for patch, color in zip(parts['boxes'], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor('#2A2F36')
        patch.set_linewidth(0.8)

    # Individualne body (jitter)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                   s=22, color='#2A2F36', alpha=0.55, zorder=3,
                   edgecolor='white', linewidth=0.4)

    ax.set_xticks(np.arange(1, len(method_order) + 1), method_order, rotation=20, ha='right')
    ax.set_ylabel('AUC priemer (cez 5 seedov)')
    ax.set_ylim(0.45, 0.95)
    ax.axhline(0.5, color='#9CA3AF', linewidth=0.8, linestyle='--', alpha=0.7)
    _header(fig,
            'Distribucia AUC pre kazdu metodu cez 6 pairwise uloh',
            'Box: kvartily; ciara: median; diamant: priemer; bodky: jednotlive pary.')
    _save(fig, '07_method_distribution', top=0.88)


# ----------------------------------------------------------------------------
# G8 — radarovy / multimetric prehlad pre top model na kazdej cielovej domene
# (multi-source, akusticke priznaky)
# ----------------------------------------------------------------------------

def plot_multisource_metrics() -> None:
    df = pd.read_csv(REPORT_DIR / '02_multisource_best.csv').copy()
    df = df.sort_values('auc_mean', ascending=True).reset_index(drop=True)

    metrics = ['auc_mean', 'f1_mean', 'sensitivity_mean', 'specificity_mean']
    metric_labels = ['AUC', 'F1', 'Sensitivity', 'Specificity']
    metric_colors = [COLOR_BASELINE, COLOR_CLASSICAL, COLOR_DEEP, COLOR_ADVERSARIAL]

    n = len(df)
    fig, ax = plt.subplots(figsize=(10.6, 0.65 * n + 2.0))
    y = np.arange(n)
    height = 0.18
    offsets = np.linspace(-1.5, 1.5, len(metrics)) * height

    for off, metric, label, color in zip(
        offsets, metrics, metric_labels, metric_colors, strict=True,
    ):
        ax.barh(y + off, df[metric], height, label=label, color=color,
                edgecolor='white', linewidth=0.4, alpha=0.92)

    ax.set_yticks(y, df['target'] + '  (' + df['method'] + ')')
    ax.set_xlabel('Hodnota metriky')
    ax.set_xlim(0.0, 1.0)
    _header(fig,
            'Multi-source: viackriterove porovnanie najlepsich modelov',
            'Pre kazdu cielovu domenu zobrazujeme AUC, F1, senzitivitu a specificitu najlepsieho modelu.')
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.08))
    ax.axvline(0.5, color='#9CA3AF', linewidth=0.8, linestyle='--', alpha=0.7)
    _save(fig, '08_multisource_multi_metric', top=0.86)


# ----------------------------------------------------------------------------
# G9 — wav2vec2 embeddingy vs akusticke priznaky (multi-source)
# ----------------------------------------------------------------------------

def plot_wav2vec2_vs_acoustic() -> None:
    """Porovnanie wav2vec2 frozen embeddingov vs rucnych akustickych priznakov v multi-source.

    Pouziva pretrained_summary.csv z PRETRAINED_DIR. Ak adresar neexistuje,
    funkcia sa tichо preskoci.
    """
    if not PRETRAINED_DIR.exists():
        print(f'[SKIP] {PRETRAINED_DIR} not found — preskakujem G9')
        return

    summary = pd.read_csv(PRETRAINED_DIR / 'pretrained_summary.csv')
    ms = summary[summary['source'].str.contains('+', regex=False)].copy()
    w2v_best = (
        ms.loc[ms.groupby('target')['auc_mean'].idxmax()]
        [['target', 'method', 'auc_mean']]
        .rename(columns={'method': 'wav2vec2_method', 'auc_mean': 'wav2vec2_auc'})
        .reset_index(drop=True)
    )

    tab = (
        pd.read_csv(REPORT_DIR / '02_multisource_best.csv')
        [['target', 'method', 'auc_mean']]
        .rename(columns={'method': 'acoustic_method', 'auc_mean': 'acoustic_auc'})
    )

    df = pd.merge(tab, w2v_best, on='target').sort_values('target').reset_index(drop=True)
    df['diff'] = df['wav2vec2_auc'] - df['acoustic_auc']
    df.to_csv(REPORT_DIR / '09_wav2vec2_vs_acoustic.csv', index=False)

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    x = np.arange(len(df))
    width = 0.36

    ax.bar(x - width / 2, df['acoustic_auc'], width,
           label='Akusticke priznaky (47-dim)', color=COLOR_BASELINE,
           edgecolor='white', linewidth=0.6)
    ax.bar(x + width / 2, df['wav2vec2_auc'], width,
           label='wav2vec2 embeddingy (1536-dim)', color=COLOR_ADVERSARIAL,
           edgecolor='white', linewidth=0.6)

    ax.set_xticks(x, df['target'])
    ymax = float(df[['acoustic_auc', 'wav2vec2_auc']].max().max()) + 0.14
    ax.set_ylim(0.50, min(1.0, ymax))
    ax.set_ylabel('AUC')
    ax.axhline(0.5, color='#9CA3AF', linewidth=0.8, linestyle='--', alpha=0.7)

    _header(fig,
            'wav2vec2 embeddingy vs akusticke priznaky: multi-source',
            'Cielova domena na osi x; obe metody trenovane v multi-source rezime (leave-one-domain-out).')
    ax.legend(loc='upper right')

    for i, row in df.iterrows():
        ax.text(i - width / 2, row['acoustic_auc'] + 0.008,
                f"{row['acoustic_auc']:.3f}", ha='center', fontsize=9, color='#1F2937')
        ax.text(i + width / 2, row['wav2vec2_auc'] + 0.008,
                f"{row['wav2vec2_auc']:.3f}", ha='center', fontsize=9, color='#1F2937')
        diff = float(row['diff'])
        ax.text(i, max(row['acoustic_auc'], row['wav2vec2_auc']) + 0.052,
                f"$\\Delta$={diff:+.3f}", ha='center', fontsize=9,
                color=(COLOR_DEEP if diff > 0.01 else ('#B5413B' if diff < -0.01 else COLOR_TEXT_DIM)))

    _save(fig, '09_wav2vec2_vs_acoustic_auc', top=0.88)
    print(f'  09_wav2vec2_vs_acoustic_auc saved ({len(df)} domains)')


# ----------------------------------------------------------------------------
# Cleanup: zmaz stare hand-coded SVG aby ostali iba matplotlib PDF/PNG
# ----------------------------------------------------------------------------
def remove_legacy_svgs() -> None:
    for svg in REPORT_DIR.glob('*.svg'):
        with suppress(OSError):
            svg.unlink()


def main() -> None:
    _style()
    plot_pairwise_best()
    plot_multisource_best()
    plot_spectrogram_best()
    plot_pairwise_heatmap()
    plot_domain_gap()
    plot_cnn_vs_acoustic()
    plot_method_distribution()
    plot_multisource_metrics()
    plot_wav2vec2_vs_acoustic()
    remove_legacy_svgs()
    print(f'Grafy ulozene do: {REPORT_DIR.resolve()}')


if __name__ == '__main__':
    main()
