"""
Pomocné funkcie - vizualizácia a utility.
Vzor: Cvičenie 2 - matplotlib vizualizácia
"""

import matplotlib

matplotlib.use('Agg')  # non-interactive backend — all output goes to files
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.config import RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """Nastavenie seedu pre reprodukovateľnosť."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def plot_losses(losses_dict, title="Trénovacie straty", save_path=None):
    """
    Vykreslenie trénovacích strát.
    Vzor: matplotlib z Cvičenie 2

    Args:
        losses_dict: dict {'názov': [loss1, loss2, ...]}
        title: nadpis grafu
        save_path: cesta na uloženie (voliteľné)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ľavý graf: všetky modely
    for name, losses in losses_dict.items():
        axes[0].plot(losses, label=name)
    axes[0].set_xlabel('Epocha')
    axes[0].set_ylabel('Strata (Loss)')
    axes[0].set_title(f'{title} (všetky)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Pravý graf: priblíženie bez DA metód (iný rozsah strát)
    da_keywords = ('DANN', 'MMD')
    for name, losses in losses_dict.items():
        if not any(k in name for k in da_keywords):
            axes[1].plot(losses, label=name)
    axes[1].set_xlabel('Epocha')
    axes[1].set_ylabel('Strata (Loss)')
    axes[1].set_title(f'{title} (in-domain priblíženie)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graf uložený: {save_path}")
    plt.close()


def plot_comparison(results, save_path=None):
    """
    Bar chart porovnanie metrík pre rôzne metódy.
    Rozdelené na 3 skupiny: In-domain A, In-domain B, Cross-domain.

    Args:
        results: dict of dicts s metrikami
        save_path: cesta na uloženie (voliteľné)
    """
    metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
    labels = ['Accuracy', 'F1', 'AUC', 'Senzitivita', 'Špecificita']

    # Rozdelenie na skupiny pre prehľadnosť
    groups = {
        'In-domain A': {k: v for k, v in results.items() if 'in-domain A' in k},
        'In-domain B': {k: v for k, v in results.items() if 'in-domain B' in k},
        'Cross-domain (A→B)': {k: v for k, v in results.items()
                               if 'A→B' in k or 'DANN' in k or 'MMD' in k},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (group_name, group_results) in zip(axes, groups.items(), strict=True):
        methods = list(group_results.keys())
        if not methods:
            continue

        x = np.arange(len(labels))
        width = 0.8 / max(len(methods), 1)

        for i, method in enumerate(methods):
            values = [group_results[method][m] for m in metrics]
            short_name = method.replace(' in-domain A', '').replace(' in-domain B', '')
            short_name = short_name.replace(' A→B', '').replace(' (baseline)', ' base')
            offset = (i - len(methods) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=short_name)

        ax.set_title(group_name, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('Hodnota')
    plt.suptitle('Porovnanie domain adaptation metód', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graf uložený: {save_path}")
    plt.close()


# ========================================================================
# ROC krivky — všetky metódy na jednom grafe
# ========================================================================

def plot_roc_all_methods(roc_data_dict, title="ROC krivky", save_path=None):
    """
    Vykreslí ROC krivky für všetky metódy na jednom grafe.

    Args:
        roc_data_dict: dict {'Metóda': (fpr, tpr, auc_score), ...}
        title: nadpis grafu
        save_path: cesta na uloženie (voliteľné)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    for (name, (fpr, tpr, auc_score)), color in zip(
        roc_data_dict.items(), colors, strict=False,
    ):
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {auc_score:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Náhodný klasifikátor')
    ax.set_xlabel('False Positive Rate (1 - Špecificita)')
    ax.set_ylabel('True Positive Rate (Senzitivita)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graf uložený: {save_path}")
    plt.close()


# ========================================================================
# t-SNE vizualizácia feature space
# ========================================================================

def plot_tsne(features_dict, labels_dict, title="t-SNE Feature Space",
              save_path=None):
    """
    t-SNE vizualizácia feature priestoru — pred a po doménovej adaptácii.

    Args:
        features_dict: dict {'Pred DA': (feats_src, feats_tgt), 'Po DA': (...)}
        labels_dict:   dict {'Pred DA': (labs_src, labs_tgt), 'Po DA': (...)}
        title: nadpis
        save_path: cesta na uloženie
    """
    n_panels = len(features_dict)
    fig, axes = plt.subplots(2, n_panels, figsize=(6 * n_panels, 10))
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    domain_colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3']
    class_colors = ['#74c476', '#fd8d3c']  # Healthy=zelená, PD=oranžová

    for col, (panel_name, (feats_src, feats_tgt)) in enumerate(features_dict.items()):
        labs_src, labs_tgt = labels_dict[panel_name]

        all_feats = np.vstack([feats_src, feats_tgt])
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                    perplexity=min(30, len(all_feats) - 1))
        embedded = tsne.fit_transform(all_feats)

        n_src = len(feats_src)
        emb_src = embedded[:n_src]
        emb_tgt = embedded[n_src:]

        # Horný rad: farba podľa DOMÉNY
        ax_top = axes[0, col]
        ax_top.scatter(emb_src[:, 0], emb_src[:, 1],
                       c=domain_colors[0], alpha=0.6, s=20, label='Zdrojová doména')
        ax_top.scatter(emb_tgt[:, 0], emb_tgt[:, 1],
                       c=domain_colors[1], alpha=0.6, s=20, label='Cieľová doména',
                       marker='^')
        ax_top.set_title(f'{panel_name}\n(farba = doména)', fontsize=10)
        ax_top.legend(fontsize=8)
        ax_top.axis('off')

        # Dolný rad: farba podľa TRIEDY (PD/Healthy)
        ax_bot = axes[1, col]
        all_labs = np.concatenate([labs_src, labs_tgt])
        for i, (x, y) in enumerate(embedded):
            c = class_colors[int(all_labs[i])]
            m = 'o' if i < n_src else '^'
            ax_bot.scatter(x, y, c=c, alpha=0.6, s=20, marker=m)

        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=class_colors[0], markersize=8, label='Zdravý'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=class_colors[1], markersize=8, label='PD'),
            Line2D([0], [0], marker='o', color='gray', markersize=8, label='Zdroj'),
            Line2D([0], [0], marker='^', color='gray', markersize=8, label='Cieľ'),
        ]
        ax_bot.legend(handles=legend_elems, fontsize=8)
        ax_bot.set_title(f'{panel_name}\n(farba = trieda)', fontsize=10)
        ax_bot.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graf uložený: {save_path}")
    plt.close()


# ========================================================================
# Performance drop — in-domain vs cross-domain
# ========================================================================

def plot_performance_drop(results_in_domain, results_cross_domain,
                          metric='auc', title="In-domain vs Cross-domain AUC",
                          save_path=None):
    """
    Vizualizuje pokles výkonu pri prechode na cross-domain scenár.

    Args:
        results_in_domain:    dict {'Metóda': {'auc': ..., ...}}  (in-domain)
        results_cross_domain: dict {'Metóda': {'auc': ..., ...}}  (cross-domain)
        metric: metrika na zobrazenie ('auc', 'f1', 'accuracy')
        title: nadpis
        save_path: cesta na uloženie
    """
    methods = list(results_in_domain.keys())
    in_vals = [results_in_domain[m].get(metric, 0) for m in methods]
    cross_vals = [results_cross_domain.get(m, {}).get(metric, 0) for m in methods]
    drops = [iv - cv for iv, cv in zip(in_vals, cross_vals, strict=True)]

    x = np.arange(len(methods))
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - width / 2, in_vals, width, label='In-domain', color='#4daf4a', alpha=0.85)
    ax1.bar(x + width / 2, cross_vals, width, label='Cross-domain', color='#e41a1c',
            alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel(metric.upper())
    ax1.set_ylim(0, 1.1)
    ax1.set_title('In-domain vs Cross-domain', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    colors_drop = ['#d73027' if d > 0 else '#1a9850' for d in drops]
    ax2.bar(x, drops, color=colors_drop, alpha=0.85)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel(f'Pokles {metric.upper()} (in-domain − cross-domain)')
    ax2.set_title('Domain Gap (červená = pokles, zelená = zisk)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graf uložený: {save_path}")
    plt.close()
