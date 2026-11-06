"""
Subspace Alignment (Fernando et al., 2013).

Ref: "Unsupervised Visual Domain Adaptation Using Subspace Alignment"
     ICCV 2013, https://arxiv.org/abs/1409.5241

Princíp (closed-form, bez tréningu NN):
  1. Vykonáme PCA na zdrojovej a cieľovej doméne nezávisle (d komponentov).
  2. Source PCA bázu X_S transformujeme do cieľovej PCA bázy X_T:
        M = X_S^T X_T   (d × d)
        X_a = X_S · M    — zarovnané source komponenty
  3. Klasifikátor (SVM, MLP) trénujeme na Source @ X_a a testujeme na Target @ X_T.

Výhoda oproti hlbokým DA metódam:
  - bez stochasticity / seed-závislosti (deterministický výsledok),
  - minutu rýchlejšie (millisekúndy),
  - často dosahuje porovnateľné alebo lepšie AUC na malých tabuľkových dátach.

Klasifikátor použitý: scikit-learn SVC s RBF jadrom + class_weight='balanced'.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SubspaceAlignmentDA:
    """Closed-form subspace alignment + RBF-SVM klasifikátor."""

    def __init__(self, n_components: int = 8, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler_s = StandardScaler()
        self.scaler_t = StandardScaler()
        self.pca_s = PCA(n_components=n_components, random_state=random_state)
        self.pca_t = PCA(n_components=n_components, random_state=random_state)
        self.M: np.ndarray | None = None
        self.clf: SVC | None = None

    def fit(self, X_source: np.ndarray, y_source: np.ndarray,
            X_target_unlabeled: np.ndarray) -> SubspaceAlignmentDA:
        # Per-domain z-score
        Xs = self.scaler_s.fit_transform(X_source)
        Xt = self.scaler_t.fit_transform(X_target_unlabeled)

        # PCA pre obe domény
        self.pca_s.fit(Xs)
        self.pca_t.fit(Xt)

        # Zarovnávacia matica M = X_S^T X_T
        # X_S = self.pca_s.components_.T (d_orig × d), X_T = self.pca_t.components_.T
        self.M = self.pca_s.components_ @ self.pca_t.components_.T  # (d × d)

        # Source projektovaný + zarovnaný do cieľovej bázy
        Xs_proj = Xs @ self.pca_s.components_.T  # (n_s, d)
        Xs_aligned = Xs_proj @ self.M             # (n_s, d) v cieľovej PCA báze

        # Klasifikátor — RBF SVM s vyváženými triedami
        self.clf = SVC(kernel="rbf", probability=True,
                       class_weight="balanced", random_state=self.random_state)
        self.clf.fit(Xs_aligned, y_source)
        return self

    def predict_proba(self, X_target: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model nie je natrénovaný. Zavolaj .fit() najprv.")
        Xt = self.scaler_t.transform(X_target)
        Xt_proj = Xt @ self.pca_t.components_.T
        return self.clf.predict_proba(Xt_proj)

    def predict(self, X_target: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_target)[:, 1] >= 0.5
