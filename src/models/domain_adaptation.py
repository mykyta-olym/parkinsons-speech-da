"""
Týždeň 7-8 + rozšírenia: Domain Adaptation techniky.

Implementácia:
  1. GRL + DANN  — Ganin & Lempitsky (2015), adversariálna metóda
  2. MMD         — Gretton et al. (2012), moment matching 1. rádu (kernel)
  3. CORAL       — Sun & Saenko (2016), moment matching 2. rádu (kovariancia)
  4. CDAN        — Long et al. (2018), class-conditional adversariálna metóda
  5. Contrastive — Yang et al. (2021), prototype-based cross-domain alignment

Spoločná architektúra:
  Feature Extractor (zdieľaný) → Label Predictor (klasifikácia PD/Healthy)
                               ↘ Domain Head (rôzny pre každú metódu)

BatchNorm v feature extractor pomáha DA tým, že normalizuje doménovo-špecifický
bias v rámci každého batchu (Ioffe & Szegedy, 2015).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    CDAN_LAMBDA,
    CONTRASTIVE_TEMP,
    DANN_LAMBDA,
    DROPOUT_RATE,
    FEATURE_DIM,
    HIDDEN_SIZE,
    NUM_CLASSES,
    NUM_FEATURES,
)

# ========================================================================
# Shared Feature Extractor (spoločný základ pre všetky DA modely)
# BatchNorm pomáha redukovať doménový shift (Ganin et al., 2015; Li et al., 2017)
# ========================================================================

def _make_feature_extractor(input_size, hidden_size, feature_dim, dropout):
    """Zdieľaná architektúra feature extraktora s BatchNorm."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, feature_dim),
        nn.BatchNorm1d(feature_dim),
        nn.ReLU(),
    )


# ========================================================================
# 1. DANN — Domain-Adversarial Neural Network
# Ref: Ganin & Lempitsky (2015), https://arxiv.org/abs/1409.7495
# ========================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Vlastná autograd funkcia pre reverziu gradientov (GRL).
    Forward: identita
    Backward: grad * (-lambda) → núti feature extractor byť doménovo-invariantný
    https://pytorch.org/docs/stable/autograd.html#function
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """GRL modul — obalenie GradientReversalFunction do nn.Module."""
    def __init__(self, lambda_val=1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class DANNModel(nn.Module):
    """
    Domain-Adversarial Neural Network.
    Architektúra: Feature Extractor → [Label Predictor]
                                    ↘ [GRL → Domain Classifier]
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                 num_domains=2, lambda_val=DANN_LAMBDA, dropout=DROPOUT_RATE):
        super().__init__()
        self.feature_extractor = _make_feature_extractor(
            input_size, hidden_size, feature_dim, dropout)
        self.label_predictor = nn.Linear(feature_dim, num_classes)
        self.grl = GradientReversalLayer(lambda_val)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_domains),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.label_predictor(features)
        domain_output = self.domain_classifier(self.grl(features))
        return class_output, domain_output, features


# ========================================================================
# 2. MMD — Maximum Mean Discrepancy
# Ref: Gretton et al. (2012), https://jmlr.org/papers/v13/gretton12a.html
# ========================================================================

def mmd_loss(source_features, target_features):
    """
    Multi-kernel MMD s RBF jadrami (median heuristic bandwidth).
    MMD² = E[k(xs,xs')] + E[k(xt,xt')] - 2·E[k(xs,xt)]
    """
    n_s = source_features.size(0)
    total = torch.cat([source_features, target_features], dim=0)
    dist_sq = torch.cdist(total, total, p=2).pow(2)

    with torch.no_grad():
        median_sq = dist_sq.median().clamp(min=1e-8)

    kernel_sum = torch.zeros_like(dist_sq)
    for factor in [0.25, 0.5, 1.0, 2.0, 4.0]:
        bandwidth = 2.0 * median_sq * factor
        kernel_sum = kernel_sum + torch.exp(-dist_sq / bandwidth.clamp(min=1e-8))

    xx = kernel_sum[:n_s, :n_s].mean()
    yy = kernel_sum[n_s:, n_s:].mean()
    xy = kernel_sum[:n_s, n_s:].mean()
    return xx + yy - 2 * xy


class MMDModel(nn.Module):
    """Klasifikátor s MMD domain adaptation."""
    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                 dropout=DROPOUT_RATE):
        super().__init__()
        self.feature_extractor = _make_feature_extractor(
            input_size, hidden_size, feature_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features


# ========================================================================
# 3. CORAL — CORrelation ALignment (Deep CORAL)
# Ref: Sun & Saenko (2016), https://arxiv.org/abs/1607.01719
#
# Minimalizuje L2-vzdialenosť medzi kovariančnými maticami zdrojovej
# a cieľovej domény — zarovná 2. štatistický moment distribúcií.
# Výhody oproti MMD: jednoduchší gradient, bez výberu kernelu, rýchlejší.
#
# L_CORAL = ||C_S - C_T||²_F / (4d²)
# ========================================================================

def coral_loss(source_features, target_features):
    """
    Deep CORAL loss: Frobenius norma rozdielov kovariančných matíc.
    Ref: Sun & Saenko (2016), Eq. 2
    """
    d = source_features.size(1)
    ns = source_features.size(0)
    nt = target_features.size(0)

    # Centrovanie (odčítanie priemeru)
    xm_s = source_features - source_features.mean(0, keepdim=True)
    xm_t = target_features - target_features.mean(0, keepdim=True)

    # Kovariančné matice
    factor_s = 1.0 / max(ns - 1, 1)
    factor_t = 1.0 / max(nt - 1, 1)
    cov_s = factor_s * (xm_s.t() @ xm_s)
    cov_t = factor_t * (xm_t.t() @ xm_t)

    # Frobenius norma rozdielu, normalizovaná podľa Sun & Saenko (2016)
    loss = (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)
    return loss


class CORALModel(nn.Module):
    """
    Klasifikátor s CORAL domain adaptation.
    Rovnaká architektúra ako MMDModel, trénovanie používa coral_loss().
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                 dropout=DROPOUT_RATE):
        super().__init__()
        self.feature_extractor = _make_feature_extractor(
            input_size, hidden_size, feature_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features


# ========================================================================
# 4. CDAN — Conditional Adversarial Domain Adaptation
# Ref: Long et al. (2018) NeurIPS, https://arxiv.org/abs/1705.10667
#
# Rozšírenie DANN: domain discriminator dostáva multilineárnu kombináciu
# features ⊗ class_predictions namiesto len features.
# Toto robí doménovú adaptáciu class-conditional — efektívnejšia ako DANN.
#
# Input doménového klasifikátora: (batch, feature_dim × num_classes)
# ========================================================================

class CDANModel(nn.Module):
    """
    Conditional Adversarial Domain Adaptation.
    Domain discriminator pracuje s: features ⊗ softmax(class_logits)
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                 num_domains=2, lambda_val=CDAN_LAMBDA, dropout=DROPOUT_RATE):
        super().__init__()
        self.feature_extractor = _make_feature_extractor(
            input_size, hidden_size, feature_dim, dropout)
        self.label_predictor = nn.Linear(feature_dim, num_classes)
        self.grl = GradientReversalLayer(lambda_val)

        # Vstup: feature_dim × num_classes (multilineárna podmienka)
        cdan_input_dim = feature_dim * num_classes
        self.domain_classifier = nn.Sequential(
            nn.Linear(cdan_input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_domains),
        )

    def forward(self, x):
        features = self.feature_extractor(x)           # (B, feature_dim)
        class_logits = self.label_predictor(features)  # (B, num_classes)
        class_probs = torch.softmax(class_logits, dim=1)  # (B, num_classes)

        # Multilineárna podmienka: vonkajší súčin features × class_probs
        # (B, feature_dim, 1) × (B, 1, num_classes) → (B, feature_dim × num_classes)
        batch_size = features.size(0)
        multilinear = torch.bmm(
            features.unsqueeze(2),
            class_probs.unsqueeze(1)
        ).view(batch_size, -1)

        domain_output = self.domain_classifier(self.grl(multilinear))
        return class_logits, domain_output, features


# ========================================================================
# 5. Contrastive Domain Adaptation
# Ref: Yang et al. (2021) CVPR, "Cross-domain Contrastive Learning for UDA"
#      He et al. (2020) "Momentum Contrast for Unsupervised Visual Representation"
#
# Princíp: prototype-based contrastive alignment medzi doménami
# - Source prototypy: priemer features pre každú triedu (Healthy, PD)
# - Target pseudo-labely: z aktuálnych predikciách modelu
# - NT-Xent strata: pull target k správnemu prototypu, push od nesprávneho
#
# L = -log( exp(sim(z_t, p_c+) / τ) / Σ exp(sim(z_t, p_c) / τ) )
# kde z_t = target feature, p_c = class prototype, τ = temperature
# ========================================================================

def prototype_contrastive_loss(source_features, source_labels,
                                target_features, target_pseudo_labels,
                                temperature=CONTRASTIVE_TEMP):
    """
    Prototype-based cross-domain contrastive alignment.

    1. Vypočíta source prototypy (priemer features per trieda)
    2. Pre každý target sample: maximalizuj kosínusovú podobnosť k správnemu prototypu
    3. NT-Xent loss s teplotou τ (temperature scaling)

    Args:
        source_features: (N_s, D) features zdrojovej domény
        source_labels:   (N_s,) labely zdrojovej domény
        target_features: (N_t, D) features cieľovej domény (len confident)
        target_pseudo_labels: (N_t,) pseudo-labely cieľovej domény
        temperature: τ pre NT-Xent (nižšie = ostrejšie rozdelenie)

    Returns:
        Skalárna contrastive loss hodnota
    """
    device = source_features.device
    num_classes = 2

    # 1. Výpočet source prototypov
    prototypes = []
    for c in range(num_classes):
        mask = (source_labels == c)
        if mask.sum() > 0:
            proto = source_features[mask].mean(0)
        else:
            # Fallback: použijeme globálny priemer (edge case pri malom datasete)
            proto = source_features.mean(0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes, dim=0)  # (num_classes, D)

    # 2. L2-normalizácia (kosínusová podobnosť)
    target_norm = F.normalize(target_features, p=2, dim=1)    # (N_t, D)
    proto_norm = F.normalize(prototypes, p=2, dim=1)           # (num_classes, D)

    # 3. Matica podobností (N_t, num_classes) / τ
    similarities = torch.mm(target_norm, proto_norm.t()) / temperature

    # 4. NT-Xent: cross-entropy kde "trieda" je správny prototyp
    target_pseudo_labels = target_pseudo_labels.to(device)
    loss = F.cross_entropy(similarities, target_pseudo_labels)
    return loss


class ContrastiveDAModel(nn.Module):
    """
    Contrastive Domain Adaptation model.
    Architektúra: Feature Extractor + Classifier (bez doménového diskriminátora)
    DA je realizovaná pomocou prototype_contrastive_loss pri trénovaní.
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                 dropout=DROPOUT_RATE):
        super().__init__()
        self.feature_extractor = _make_feature_extractor(
            input_size, hidden_size, feature_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features
