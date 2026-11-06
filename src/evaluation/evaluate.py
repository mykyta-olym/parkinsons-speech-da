"""
Týždeň 6: Evaluácia modelov.
Vzor: Cvičenie 4 - FF.ipynb (evaluácia s torch.no_grad)

Metriky pre klinické nasadenie:
  - AUC (Area Under ROC Curve)
  - F1 skóre
  - Senzitivita (recall) - schopnosť zachytiť PD pacientov
  - Špecificita - schopnosť správne identifikovať zdravých

Ref: sklearn metriky
https://scikit-learn.org/stable/modules/model_evaluation.html
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.config import DEVICE, N_BOOTSTRAP


def _model_forward(model, features, model_type):
    """
    Unifikovaný forward pass — vráti (class_logits, features_out).
    'dann' a 'cdan' vracajú (logits, domain, feats) → extrakt prvý a tretí.
    'mmd', 'coral', 'contrastive' vracajú (logits, feats).
    'standard' vracia len logits → features_out = None.
    """
    if model_type in ('dann', 'cdan'):
        logits, _, feats = model(features)
    elif model_type in ('mmd', 'coral', 'contrastive'):
        logits, feats = model(features)
    else:
        logits = model(features)
        feats = None
    return logits, feats


def evaluate_model(model, test_loader, model_type='standard'):
    """
    Evaluácia modelu na testovacej sade.
    Vzor: Cvičenie 4 - with torch.no_grad(): ...

    Args:
        model: natrénovaný PyTorch model
        test_loader: DataLoader s testovacími dátami
        model_type: 'standard' (MLP/CNN) alebo 'dann' alebo 'mmd'

    Returns:
        dict s metrikami (accuracy, f1, auc, sensitivity, specificity)
    """
    model = model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    # Vzor z Cvičenia 4: evaluácia bez gradientov
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)

            # Forward pass podľa typu modelu
            outputs, _ = _model_forward(model, features, model_type)

            # Pravdepodobnosti a predikcie
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Výpočet metrík
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # AUC - vyžaduje obe triedy v dátach
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    # Senzitivita a špecificita z konfúznej matice
    if len(np.unique(all_labels)) == 2:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0

    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def evaluate_svm(svm_model, X_test, y_test):
    """
    Evaluácia SVM modelu (scikit-learn).

    Args:
        svm_model: natrénovaný Pipeline (scaler + SVM)
        X_test: testovacie príznaky
        y_test: testovacie labely
    """
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0

    if len(np.unique(y_test)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0

    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def print_metrics(name, metrics):
    """Prehľadný výpis metrík."""
    print(f"  {name}:")
    print(f"    Accuracy:    {metrics['accuracy']:.3f}")
    print(f"    F1:          {metrics['f1']:.3f}")
    print(f"    AUC:         {metrics['auc']:.3f}")
    print(f"    Senzitivita: {metrics['sensitivity']:.3f}")
    print(f"    Špecificita: {metrics['specificity']:.3f}")


def print_comparison_table(results):
    """
    Tabuľka porovnania výsledkov.

    Args:
        results: dict of dicts, napr. {'MLP in-domain': {...}, 'DANN': {...}}
    """
    print("\n" + "=" * 75)
    print("POROVNANIE VÝSLEDKOV")
    print("=" * 75)
    header = f"{'Metóda':<30} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Sens':>6} {'Spec':>6}"
    print(header)
    print("-" * 75)

    for name, m in results.items():
        row = (f"{name:<30} "
               f"{m['accuracy']:>6.3f} "
               f"{m['f1']:>6.3f} "
               f"{m['auc']:>6.3f} "
               f"{m['sensitivity']:>6.3f} "
               f"{m['specificity']:>6.3f}")
        print(row)

    print("=" * 75)


# ========================================================================
# ROC dáta pre vizualizáciu
# ========================================================================

def get_roc_data(model, test_loader, model_type='standard'):
    """
    Vráti (fpr, tpr, auc_score) pre vykreslenie ROC krivky.

    Args:
        model: natrénovaný PyTorch model
        test_loader: DataLoader s testovacími dátami
        model_type: 'standard' | 'dann' | 'mmd' | 'coral' | 'cdan' | 'contrastive'

    Returns:
        tuple (fpr, tpr, auc_score) — numpy polia
    """
    model = model.to(DEVICE)
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            outputs, _ = _model_forward(model, features, model_type)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        fpr, tpr, auc_score = np.array([0, 1]), np.array([0, 1]), 0.5

    return fpr, tpr, auc_score


# ========================================================================
# Extrakcia príznakov pre t-SNE vizualizáciu
# ========================================================================

def extract_features(model, loader, model_type='standard'):
    """
    Extrahuje príznaky z feature_extractor vrstvy pre t-SNE vizualizáciu.

    Returns:
        (features_np, labels_np) — numpy polia
    """
    model = model.to(DEVICE)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            _, feats = _model_forward(model, features, model_type)
            if feats is not None:
                all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())

    if not all_features:
        return None, np.array(all_labels)

    return np.vstack(all_features), np.array(all_labels)


# ========================================================================
# Predikcie (y_true, y_score) — pre bootstrap a ďalšie analýzy
# ========================================================================

def get_predictions(model, loader, model_type='standard'):
    """
    Vráti (y_true, y_score) — skutočné labely a pravdepodobnosti pre PD triedu.
    Použitie: bootstrap CI, vlastné analýzy.
    """
    model = model.to(DEVICE)
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            outputs, _ = _model_forward(model, features, model_type)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


# ========================================================================
# Bootstrap confidence interval pre AUC
# ========================================================================

def bootstrap_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP, alpha=0.95,
                 random_state=42):
    """
    Bootstrap confidence interval pre AUC.

    Args:
        y_true: skutočné labely (0/1)
        y_score: skóre pre pozitívnu triedu (pravdepodobnosť PD)
        n_bootstrap: počet bootstrap iterácií
        alpha: úroveň spoľahlivosti (0.95 → 95% CI)
        random_state: seed pre reprodukovateľnosť

    Returns:
        (ci_low, ci_high): hranice intervalu spoľahlivosti
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    auc_scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        y_true_b = y_true[idx]
        y_score_b = y_score[idx]

        if len(np.unique(y_true_b)) < 2:
            continue

        try:
            auc_b = roc_auc_score(y_true_b, y_score_b)
            auc_scores.append(auc_b)
        except ValueError:
            continue

    if not auc_scores:
        return 0.0, 0.0

    lower = (1.0 - alpha) / 2.0
    upper = 1.0 - lower
    return float(np.quantile(auc_scores, lower)), float(np.quantile(auc_scores, upper))


# ========================================================================
# Paired bootstrap pre rozdiel AUC (Δ = method - baseline)
#
# Toto je SPRÁVNY štatistický test pre porovnanie dvoch klasifikátorov
# na rovnakom testovacom sete. Bootstrap nezávisle nadhodnocuje variance,
# pretože ignoruje koreláciu medzi predikciami modelov na tých istých
# vzorkách. Paired bootstrap používa rovnaké indexy pre obe metódy
# v každej iterácii → CI na Δ je výrazne tesnejší.
#
# Ref: Hanley & McNeil (1983), DeLong et al. (1988); paired bootstrap je
# distribution-free alternatíva s totožnými asymptotickými vlastnosťami.
# ========================================================================

def paired_bootstrap_diff(y_true, y_score_a, y_score_b,
                          n_bootstrap=N_BOOTSTRAP, alpha=0.95,
                          random_state=42):
    """Paired bootstrap CI pre Δ AUC = AUC(b) - AUC(a) na rovnakom test sete.

    Returns (delta_point, ci_low, ci_high, p_value_two_sided).
    p-value: podiel bootstrap iterácií, kde Δ má opačné znamienko ako bod
    odhad — empirická one-sided pravdepodobnosť, zdvojnásobená pre two-sided.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    delta_point = roc_auc_score(y_true, y_score_b) - roc_auc_score(y_true, y_score_a)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            d = (roc_auc_score(yt, y_score_b[idx]) -
                 roc_auc_score(yt, y_score_a[idx]))
            diffs.append(d)
        except ValueError:
            continue

    if not diffs:
        return float(delta_point), 0.0, 0.0, 1.0

    diffs_arr = np.asarray(diffs)
    lower = (1.0 - alpha) / 2.0
    upper = 1.0 - lower
    ci_low = float(np.quantile(diffs_arr, lower))
    ci_high = float(np.quantile(diffs_arr, upper))

    # Empirická p-value (two-sided, na centrovaných rozdieloch)
    centered = diffs_arr - diffs_arr.mean()
    p_two = float(np.mean(np.abs(centered) >= np.abs(delta_point)))
    return float(delta_point), ci_low, ci_high, p_two
