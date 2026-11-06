"""
Týždeň 4-8: Trénovacie a evaluačné funkcie.

Trénovací cyklus: Cvičenie 4 - FF.ipynb
  for epoch → for batch → forward → loss → backward → step

Evaluácia: Cvičenie 4 - with torch.no_grad(): ...
"""

import math
from itertools import cycle

import torch
import torch.nn as nn

from src.config import (
    CONTRASTIVE_CONF,
    CONTRASTIVE_LAMBDA,
    CONTRASTIVE_TEMP,
    CORAL_LAMBDA,
    DA_EPOCHS,
    DA_LR,
    DEVICE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MMD_LAMBDA,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)

# ========================================================================
# Trénovanie (Týždeň 4-5)
# Vzor: Cvičenie 4 - FF.ipynb, trénovací cyklus
# ========================================================================

def train_model(model, train_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                class_weights=None):
    """
    Štandardné trénovanie klasifikátora (bez domain adaptácie).
    Vzor: Cvičenie 4 - trénovací cyklus na MNIST

    Args:
        model: PyTorch model (MLP alebo CNN1D)
        train_loader: DataLoader s trénovacími dátami
        num_epochs: počet epoch
        lr: learning rate
        class_weights: váhy tried pre vyváženú loss (voliteľné)
    """
    model = model.to(DEVICE)
    model.train()

    # Loss a optimalizátor - vzor z Cvičenie 4
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    weight = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for features, labels in train_loader:
            # Presun na zariadenie - vzor z Cvičenie 4
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass - vzor z Cvičenie 4
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Výpis každých 10 epoch
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return losses


# ========================================================================
# Trénovanie DANN (Týždeň 7)
# Ref: Ganin et al. (2015), Algorithm 1
# https://arxiv.org/abs/1409.7495
# ========================================================================

def train_dann(model, source_loader, target_loader,
               num_epochs=DA_EPOCHS, lr=DA_LR, class_weights=None):
    """
    Trénovanie DANN - adversariálna domain adaptácia.

    Source loader: dáta s labelmi (PD/Healthy) zo zdrojovej domény
    Target loader: dáta BEZ labelov z cieľovej domény
                   (labely sa NEPOUŽÍVAJÚ pri trénovaní)
    """
    model = model.to(DEVICE)
    model.train()

    weight = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    domain_criterion = nn.CrossEntropyLoss()  # doménová strata bez váh
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Inverse LR schedule podľa Ganin et al. (2015), Eq. 11:
    # μ_p = μ_0 / (1 + α·p)^β, kde α=10, β=0.75
    # LR klesá ~7x počas trénovania → stabilný koniec
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + 10.0 * epoch / max(num_epochs - 1, 1)) ** 0.75
    )

    # Uložíme pôvodnú max lambda pre progresívne zvyšovanie
    max_lambda = model.grl.lambda_val

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Sigmoidný schedule podľa Ganin et al. (2015):
        # λ_p = 2 / (1 + exp(-γ·p)) - 1, kde p = epoch/epochs, γ = 10
        # Saturuje okolo 70% trénovania → stabilný koniec
        progress = epoch / max(num_epochs - 1, 1)
        current_lambda = max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
        model.grl.lambda_val = current_lambda

        # zip + cycle: iterujeme cez oba loadery súčasne
        # cycle opakuje kratší loader
        for (src_x, src_y), (tgt_x, _) in zip(source_loader, cycle(target_loader)):
            src_x = src_x.to(DEVICE)
            src_y = src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            # Konkatenovaný forward: BatchNorm vidí obe domény v jednej dávke
            # → statistiky sa počítajú zo zmiešanej distribúcie (krízovým AdaBN-style).
            n_s = src_x.size(0)
            cat_x = torch.cat([src_x, tgt_x], dim=0)
            cat_class, cat_domain, _ = model(cat_x)
            src_class = cat_class[:n_s]
            src_domain = cat_domain[:n_s]
            tgt_domain = cat_domain[n_s:]

            # Klasifikačná strata (len na source!)
            class_loss = criterion(src_class, src_y)

            # Doménová strata (na oboch doménach)
            # Source = 0, Target = 1
            src_domain_labels = torch.zeros(n_s, dtype=torch.long, device=DEVICE)
            tgt_domain_labels = torch.ones(tgt_x.size(0), dtype=torch.long,
                                           device=DEVICE)
            domain_loss = (domain_criterion(src_domain, src_domain_labels) +
                           domain_criterion(tgt_domain, tgt_domain_labels))

            # Celková strata
            # GRL v modeli sa stará o obrátenie gradientov domain_loss
            loss = class_loss + domain_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        # LR schedule step
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f} '
                  f'(lambda={current_lambda:.3f}, lr={current_lr:.6f}, '
                  f'class: {class_loss.item():.4f}, '
                  f'domain: {domain_loss.item():.4f})')

    return losses


# ========================================================================
# Trénovanie s MMD (Týždeň 8)
# Ref: Gretton et al. (2012)
# ========================================================================

def train_mmd(model, source_loader, target_loader,
              num_epochs=DA_EPOCHS, lr=DA_LR, mmd_lambda=MMD_LAMBDA,
              class_weights=None):
    """
    Trénovanie s MMD stratou.
    Klasifikačná strata + MMD strata medzi zdrojovými a cieľovými príznakmi.
    """
    from src.models.domain_adaptation import mmd_loss

    model = model.to(DEVICE)
    model.train()

    weight = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (src_x, src_y), (tgt_x, _) in zip(source_loader, cycle(target_loader)):
            src_x = src_x.to(DEVICE)
            src_y = src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            n_s = src_x.size(0)
            cat_x = torch.cat([src_x, tgt_x], dim=0)
            cat_out, cat_feats = model(cat_x)
            src_output = cat_out[:n_s]
            src_features = cat_feats[:n_s]
            tgt_features = cat_feats[n_s:]

            # Klasifikačná strata
            class_loss = criterion(src_output, src_y)

            # MMD strata - zarovnanie distribúcií
            mmd = mmd_loss(src_features, tgt_features)

            # Celková strata
            loss = class_loss + mmd_lambda * mmd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f} '
                  f'(class: {class_loss.item():.4f}, '
                  f'MMD: {mmd.item():.4f})')

    return losses


# ========================================================================
# Trénovanie s CORAL (Deep CORrelation ALignment)
# Ref: Sun & Saenko (2016), https://arxiv.org/abs/1607.01719
# Totožná slučka ako MMD, len strata nahradená coral_loss().
# ========================================================================

def train_coral(model, source_loader, target_loader,
                num_epochs=DA_EPOCHS, lr=DA_LR, coral_lambda=CORAL_LAMBDA,
                class_weights=None):
    """
    Trénovanie s CORAL stratou.
    Klasifikačná strata + CORAL strata (Frobenius norma rozdielov kovariancií).
    """
    from src.models.domain_adaptation import coral_loss

    model = model.to(DEVICE)
    model.train()

    weight = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (src_x, src_y), (tgt_x, _) in zip(source_loader, cycle(target_loader)):
            src_x = src_x.to(DEVICE)
            src_y = src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            n_s = src_x.size(0)
            cat_x = torch.cat([src_x, tgt_x], dim=0)
            cat_out, cat_feats = model(cat_x)
            src_output = cat_out[:n_s]
            src_features = cat_feats[:n_s]
            tgt_features = cat_feats[n_s:]

            class_loss = criterion(src_output, src_y)
            c_loss = coral_loss(src_features, tgt_features)
            loss = class_loss + coral_lambda * c_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f} '
                  f'(class: {class_loss.item():.4f}, '
                  f'CORAL: {c_loss.item():.4f})')

    return losses


# ========================================================================
# Trénovanie CDAN (Conditional Adversarial Domain Adaptation)
# Ref: Long et al. (2018) NeurIPS, https://arxiv.org/abs/1705.10667
# Rovnaká slučka ako DANN — CDANModel interne rieši multilineárnu podmienku.
# ========================================================================

def train_cdan(model, source_loader, target_loader,
               num_epochs=DA_EPOCHS, lr=DA_LR, class_weights=None):
    """
    Trénovanie CDAN — class-conditional adversariálna DA.
    Identická slučka ako train_dann(); multilineárne podmienkovanie
    je transparentne implementované v CDANModel.forward().
    """
    model = model.to(DEVICE)
    model.train()

    weight = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: 1.0 / (1.0 + 10.0 * ep / max(num_epochs - 1, 1)) ** 0.75
    )

    max_lambda = model.grl.lambda_val
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        progress = epoch / max(num_epochs - 1, 1)
        current_lambda = max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
        model.grl.lambda_val = current_lambda

        for (src_x, src_y), (tgt_x, _) in zip(source_loader, cycle(target_loader)):
            src_x = src_x.to(DEVICE)
            src_y = src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            n_s = src_x.size(0)
            cat_x = torch.cat([src_x, tgt_x], dim=0)
            cat_class, cat_domain, _ = model(cat_x)
            src_class = cat_class[:n_s]
            src_domain = cat_domain[:n_s]
            tgt_domain = cat_domain[n_s:]

            class_loss = criterion(src_class, src_y)

            src_domain_labels = torch.zeros(n_s, dtype=torch.long, device=DEVICE)
            tgt_domain_labels = torch.ones(tgt_x.size(0), dtype=torch.long, device=DEVICE)
            domain_loss = (domain_criterion(src_domain, src_domain_labels) +
                           domain_criterion(tgt_domain, tgt_domain_labels))

            loss = class_loss + domain_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f} '
                  f'(lambda={current_lambda:.3f}, '
                  f'class: {class_loss.item():.4f}, '
                  f'domain: {domain_loss.item():.4f})')

    return losses


# ========================================================================
# Trénovanie Contrastive DA
# Ref: Yang et al. (2021) CVPR; He et al. (2020) MoCo
#
# Source: supervised cross-entropy
# Target: prototype_contrastive_loss pre confident pseudo-labeled samples
# ========================================================================

def train_contrastive(model, source_loader, target_loader,
                      num_epochs=DA_EPOCHS, lr=DA_LR,
                      contrastive_lambda=CONTRASTIVE_LAMBDA,
                      conf_threshold=CONTRASTIVE_CONF,
                      temperature=CONTRASTIVE_TEMP,
                      class_weights=None):
    """
    Trénovanie Contrastive DA — prototype alignment s pseudo-labelmi.

    Pre každý target batch:
      1. Zrazí softmax pravdepodobnosti → pseudo-labely
      2. Odfiltruje nízko-konfidenčné vzorky (< conf_threshold)
      3. Ak ostali ≥ 2 confident vzorky: vypočíta prototype_contrastive_loss
    """
    from src.models.domain_adaptation import prototype_contrastive_loss

    model = model.to(DEVICE)
    model.train()

    weight = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (src_x, src_y), (tgt_x, _) in zip(source_loader, cycle(target_loader)):
            src_x = src_x.to(DEVICE)
            src_y = src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            # Konkatenovaný forward (BatchNorm vidí obe domény)
            n_s = src_x.size(0)
            cat_x = torch.cat([src_x, tgt_x], dim=0)
            cat_out, cat_feats = model(cat_x)
            src_output = cat_out[:n_s]
            src_features = cat_feats[:n_s]
            tgt_output = cat_out[n_s:]
            tgt_features = cat_feats[n_s:]

            # Source: supervised classification
            class_loss = criterion(src_output, src_y)

            # Target: pseudo-labels via confidence thresholding
            tgt_probs = torch.softmax(tgt_output, dim=1)
            tgt_conf, tgt_pseudo = tgt_probs.max(dim=1)

            confident_mask = tgt_conf >= conf_threshold
            loss = class_loss

            if confident_mask.sum() >= 2:
                conf_feats = tgt_features[confident_mask]
                conf_pseudo = tgt_pseudo[confident_mask]
                c_loss = prototype_contrastive_loss(
                    src_features.detach(), src_y,
                    conf_feats, conf_pseudo,
                    temperature=temperature
                )
                loss = class_loss + contrastive_lambda * c_loss
            else:
                c_loss = torch.tensor(0.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            n_conf = confident_mask.sum().item()
            print(f'  Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f} '
                  f'(class: {class_loss.item():.4f}, '
                  f'contrastive: {c_loss.item():.4f}, '
                  f'conf_samples: {n_conf})')

    return losses


# ========================================================================
# Multi-source DANN
# Loader z create_multisource_loaders() kombinuje viac zdrojových domén.
# Trénovacia slučka je identická s train_dann().
# ========================================================================

def train_multisource_dann(model, source_loader, target_loader,
                           num_epochs=DA_EPOCHS, lr=DA_LR,
                           class_weights=None):
    """
    Multi-source DANN — zdrojový loader obsahuje spojené vzorky z viacerých domén.
    Trénovacia logika je totožná s train_dann().
    """
    return train_dann(model, source_loader, target_loader,
                      num_epochs=num_epochs, lr=lr,
                      class_weights=class_weights)
