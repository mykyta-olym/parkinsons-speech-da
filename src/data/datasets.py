"""
Týždeň 2: Vlastné Dataset triedy a DataLoader.
Vzor: Cvičenie 2 - Práca s dátami.ipynb (trieda Dataset, DataLoader)

Novinky oproti pôvodnej verzii:
  - patient_wise_loaders(): správne delenie podľa pacientov (bez data leakage)
  - create_multisource_loaders(): spojenie viacerých zdrojových domén
  - load_domain(): podpora rozšírených príznakov Istanbul (47 features)
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, BATCH_SIZE_DA, DATA_DIR, RANDOM_SEED


class ParkinsonDataset(Dataset):
    """
    Vlastný Dataset pre akustické príznaky Parkinsonovej choroby.
    Vzor: Cvičenie 2 - triedy Dataset s __len__ a __getitem__
    """

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def compute_class_weights(y):
    """Výpočet váh tried – dôležité pre nevyvážené datasety (Oxford: 75% PD)."""
    classes = np.unique(y)
    n_samples = len(y)
    weights = []
    for c in classes:
        count = np.sum(y == c)
        weights.append(n_samples / (len(classes) * count))
    return torch.tensor(weights, dtype=torch.float32)


def load_domain(domain_name, extended=False):
    """
    Načíta CSV súbor pre danú doménu.

    Args:
        domain_name: 'oxford', 'istanbul', 'istanbul_r1', 'istanbul_r2', 'istanbul_r3'
        extended: ak True, načíta rozšírené príznaky Istanbul (MFCC, Delta, GNE)
                  Pozor: extended=True funguje len pre Istanbul datasety.

    Returns:
        X (np.array): príznakový vektor
        y (np.array): binárne labely (0=Healthy, 1=PD)
        patient_ids (np.array alebo None): IDs pacientov pre patient-wise split
    """
    path = os.path.join(DATA_DIR, f'{domain_name}.csv')
    df = pd.read_csv(path)

    meta_cols = {'label', 'domain', 'patient_id', 'recording'}

    if extended:
        # Základných 12 + rozšírené príznaky (ext_* prefix)
        feature_cols = [c for c in df.columns
                        if c not in meta_cols and not c.startswith('ext_')]
        ext_cols = [c for c in df.columns if c.startswith('ext_')]
        feature_cols = feature_cols + ext_cols
    else:
        # Len základných 12 príznakov (kompatibilné s Oxford) — vylúčiť ext_*
        feature_cols = [c for c in df.columns
                        if c not in meta_cols and not c.startswith('ext_')]

    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    patient_ids = df['patient_id'].values if 'patient_id' in df.columns else None

    return X, y, patient_ids


def patient_wise_loaders(X, y, patient_ids, test_size=0.25, batch_size=BATCH_SIZE):
    """
    Rozdelenie podľa pacientov namiesto náhodného rozdelenia vzoriek.

    PREČO: Keď sú viacnásobné záznamy toho istého pacienta v train aj test sade,
    model sa "naučí hlas" konkrétneho pacienta namiesto príznakov choroby
    → nerealisticky vysoké metriky (data leakage).

    Správny prístup: všetky záznamy rovnakého pacienta idú buď do train ALEBO test.

    Args:
        X: príznaky
        y: labely
        patient_ids: ID pacienta pre každú vzorku
        test_size: podiel testovacích PACIENTOV (nie vzoriek)

    Returns:
        train_loader, test_loader, scaler
    """
    # Unikátni pacienti s ich triedou (class majority vote)
    unique_patients = np.unique(patient_ids)
    patient_labels = {}
    for pid in unique_patients:
        mask = patient_ids == pid
        patient_labels[pid] = int(np.bincount(y[mask]).argmax())

    patients = list(unique_patients)
    p_labels = [patient_labels[p] for p in patients]

    # Stratifikovaný split na úrovni PACIENTOV
    train_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=RANDOM_SEED,
        stratify=p_labels
    )
    train_patients_set = set(train_patients)
    test_patients_set = set(test_patients)

    train_mask = np.array([pid in train_patients_set for pid in patient_ids])
    test_mask = np.array([pid in test_patients_set for pid in patient_ids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Normalizácia – fit len na train!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(ParkinsonDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ParkinsonDataset(X_test, y_test),
                             batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler


def create_loaders(X, y, test_size=0.2, batch_size=BATCH_SIZE):
    """
    Rozdelí náhodne (bez patient-wise), normalizuje a vytvorí DataLoader.
    Vzor: Cvičenie 2 - DataLoader s batch_size a shuffle
    Používa sa pre Istanbul sessions (každý pacient má len 1 záznam per session).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(ParkinsonDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ParkinsonDataset(X_test, y_test),
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, scaler


def create_cross_domain_loaders(X_source, y_source, X_target, y_target,
                                batch_size=BATCH_SIZE_DA):
    """
    Loadery pre cross-domain experimenty.
    Celý source = train, celý target = test.
    Každá doména normalizovaná vlastným scalerom (štandard v DA literatúre).
    """
    source_scaler = StandardScaler()
    target_scaler = StandardScaler()
    X_source_n = source_scaler.fit_transform(X_source)
    X_target_n = target_scaler.fit_transform(X_target)

    source_loader = DataLoader(ParkinsonDataset(X_source_n, y_source),
                               batch_size=batch_size, shuffle=True, drop_last=True)
    target_train_loader = DataLoader(ParkinsonDataset(X_target_n, y_target),
                                     batch_size=batch_size, shuffle=True, drop_last=True)
    # Test loader: non-shuffled, bez drop_last → presná evaluácia
    target_test_loader = DataLoader(ParkinsonDataset(X_target_n, y_target),
                                    batch_size=batch_size * 2, shuffle=False, drop_last=False)
    return source_loader, target_train_loader, target_test_loader


def create_multisource_loaders(source_list, X_target, y_target,
                               batch_size=BATCH_SIZE_DA):
    """
    Loadery pre multi-source domain adaptation.
    Kombinuje viaceré zdrojové domény do jedného trénovacieho setu.

    Princíp (Mansour et al., 2009): vážená distribúcia viacerých zdrojov
    je lepšia ako single-source, keď žiadna zdrojová doména nie je
    identická s cieľovou. Implementácia: jednoduché konkatenácia zdrojov
    (každý rebalancovaný na rovnaký počet vzoriek).

    Args:
        source_list: list of (X, y, domain_label) tuples
        X_target, y_target: cieľová doména

    Returns:
        source_loader, target_loader, target_scaler
    """
    # Rebalancovanie zdrojov (všetky na rovnaký počet vzoriek)
    min_size = min(len(X) for X, _, _ in source_list)
    X_parts, y_parts, d_parts = [], [], []
    for X_src, y_src, d_label in source_list:
        if len(X_src) > min_size:
            # náhodný výber bez opakování
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(len(X_src), min_size, replace=False)
            X_src, y_src = X_src[idx], y_src[idx]
        # Per-domain normalizácia
        scaler = StandardScaler()
        X_parts.append(scaler.fit_transform(X_src))
        y_parts.append(y_src)
        d_parts.append(np.full(len(y_src), d_label, dtype=np.int64))

    X_combined = np.concatenate(X_parts)
    y_combined = np.concatenate(y_parts)

    # Target normalizácia
    target_scaler = StandardScaler()
    X_target_n = target_scaler.fit_transform(X_target)

    source_loader = DataLoader(ParkinsonDataset(X_combined, y_combined),
                               batch_size=batch_size, shuffle=True, drop_last=True)
    target_train_loader = DataLoader(ParkinsonDataset(X_target_n, y_target),
                                     batch_size=batch_size, shuffle=True, drop_last=True)
    target_test_loader = DataLoader(ParkinsonDataset(X_target_n, y_target),
                                    batch_size=batch_size * 2, shuffle=False, drop_last=False)
    return source_loader, target_train_loader, target_test_loader
