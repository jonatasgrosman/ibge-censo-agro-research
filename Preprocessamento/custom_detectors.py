"""
custom_detectors.py — Detectores customizados compatíveis com PyOD
==================================================================
Cada classe herda de pyod.models.base.BaseDetector e expõe:
  fit(X)               → treina e calcula decision_scores_
  decision_function(X) → retorna scores de anomalia (maior = mais anômalo)

Para adicionar um novo modelo externo:
  1. Crie uma subclasse de BaseDetector aqui
  2. Implemente fit(X) e decision_function(X)
  3. Adicione uma entrada em MODELS_REGISTRY no notebook

Dependências externas:
  pip install rrcf eif
"""

import numpy as np
from pyod.models.base import BaseDetector
from sklearn.utils.validation import check_is_fitted


# ─────────────────────────────────────────────────────────────────────────────
# RCFDetector — Random Cut Forest
# ─────────────────────────────────────────────────────────────────────────────
class RCFDetector(BaseDetector):
    """
    Random Cut Forest (RCF) usando a biblioteca `rrcf`.

    Constrói uma floresta de árvores de corte aleatório e usa o
    Collusive Displacement (CoDisp) médio como score de anomalia.
    Pontos com alto CoDisp são isolados mais facilmente e portanto
    mais anômalos.

    Vantagem sobre IForest: detecta melhor anomalias localizadas em
    subespaços e é naturalmente adaptado a fluxos (streaming).

    Referência: Guha et al. (2016). Robust Random Cut Forest Based
    Anomaly Detection on Streams. ICML.

    Instalação: pip install rrcf
    """

    def __init__(self, contamination=0.1, n_trees=200, tree_size=256,
                 random_state=None):
        super().__init__(contamination=contamination)
        self.n_trees = n_trees
        self.tree_size = tree_size
        self.random_state = random_state

    def fit(self, X, y=None):
        import rrcf
        self._set_n_classes(y)
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        self.forest_ = []
        sum_codisp = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.int32)

        for _ in range(self.n_trees):
            sz = min(self.tree_size, n)
            idx = rng.choice(n, size=sz, replace=False)
            tree = rrcf.RCTree()
            for i in idx:
                tree.insert_point(X[i], index=int(i))
            for i in idx:
                sum_codisp[i] += tree.codisp(int(i))
                count[i] += 1
            self.forest_.append(tree)

        with np.errstate(invalid='ignore', divide='ignore'):
            self.decision_scores_ = np.where(count > 0, sum_codisp / count, 0.0)

        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """
        Insere cada ponto de teste temporariamente em cada árvore,
        calcula CoDisp e remove. Correto porém O(n_test × n_trees).
        Para datasets grandes prefira reduzir n_trees.
        """
        check_is_fitted(self, ['forest_'])
        import rrcf
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        scores = np.zeros(n, dtype=np.float64)
        prefix = '_infer_'

        for tree in self.forest_:
            for i in range(n):
                key = f'{prefix}{i}'
                tree.insert_point(X[i], index=key)
                scores[i] += tree.codisp(key)
                tree.forget_point(key)

        return scores / len(self.forest_)


# ─────────────────────────────────────────────────────────────────────────────
# EIFDetector — Extended Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────
class EIFDetector(BaseDetector):
    """
    Extended Isolation Forest (EIF) usando a biblioteca `eif`.

    Generaliza o IForest substituindo cortes axis-aligned por
    hiperplanos oblíquos aleatórios (extension_level controla
    o grau de obliquidade). Isso elimina o viés direcional do
    IForest padrão, produzindo scores mais isotrópicos no espaço
    de features e melhor separação em anomalias circulares/clusterizadas.

    extension_level=0  → equivalente ao IForest padrão (axis-aligned)
    extension_level=d-1 → extensão máxima (hiperplanos completamente oblíquos)

    Referência: Hariri et al. (2021). Extended Isolation Forest.
    IEEE Transactions on Knowledge and Data Engineering.

    Instalação: pip install eif
    """

    def __init__(self, contamination=0.1, n_trees=200, sample_size=256,
                 extension_level=None, random_state=None):
        super().__init__(contamination=contamination)
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.extension_level = extension_level  # None → n_features - 1
        self.random_state = random_state

    def fit(self, X, y=None):
        import eif
        self._set_n_classes(y)
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        ext = self.extension_level if self.extension_level is not None else (d - 1)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.forest_ = eif.iForest(
            X,
            ntrees=self.n_trees,
            sample_size=min(self.sample_size, n),
            ExtensionLevel=ext,
        )
        self.decision_scores_ = self.forest_.compute_paths(X_in=X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['forest_'])
        X = np.asarray(X, dtype=np.float64)
        return self.forest_.compute_paths(X_in=X)


# ─────────────────────────────────────────────────────────────────────────────
# SubspaceIForest — IForest em subespaços aleatórios de features
# ─────────────────────────────────────────────────────────────────────────────
class SubspaceIForest(BaseDetector):
    """
    Subspace Isolation Forest — variante do IForest que agrega scores
    calculados em múltiplos subespaços aleatórios de features.

    Motivação para dados do Censo (esparsos e de alta dimensão):
    Anomalias frequentemente se manifestam em combinações específicas de
    poucas variáveis (ex.: área declarada inconsistente com tipo de cultura).
    O IForest global pode perder esse sinal ao considerar todas as features
    conjuntamente. Ao avaliar subespaços menores, este modelo amplifica
    anomalias localizadas.

    Diferença do FeatureBagging (PyOD): FB usa LOF interno; aqui usamos
    IForest, preservando a eficiência em alta dimensão.

    Não requer dependências externas além de PyOD.

    Parâmetros:
      n_subspaces   : número de subespaços (default 50)
      subspace_size : tamanho de cada subespaço; None → sqrt(n_features)
      n_estimators  : árvores por subespaço IForest (default 100)
    """

    def __init__(self, contamination=0.1, n_subspaces=50, subspace_size=None,
                 n_estimators=100, random_state=None):
        super().__init__(contamination=contamination)
        self.n_subspaces = n_subspaces
        self.subspace_size = subspace_size
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):
        from pyod.models.iforest import IForest as _IForest
        self._set_n_classes(y)
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        sz = self.subspace_size or max(2, int(np.sqrt(d)))
        sz = min(sz, d)

        self.subspaces_ = []
        self.estimators_ = []

        for _ in range(self.n_subspaces):
            feat_idx = rng.choice(d, size=sz, replace=False)
            seed = int(rng.integers(0, 2**31))
            clf = _IForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=seed,
                n_jobs=-1,
            )
            clf.fit(X[:, feat_idx])
            self.subspaces_.append(feat_idx)
            self.estimators_.append(clf)

        raw = np.column_stack([clf.decision_scores_ for clf in self.estimators_])
        self.decision_scores_ = self._aggregate(raw)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['subspaces_', 'estimators_'])
        X = np.asarray(X, dtype=np.float64)
        raw = np.column_stack([
            clf.decision_function(X[:, feat_idx])
            for clf, feat_idx in zip(self.estimators_, self.subspaces_)
        ])
        return self._aggregate(raw)

    @staticmethod
    def _aggregate(scores_matrix: np.ndarray) -> np.ndarray:
        """Normaliza cada subespaço para [0,1] e agrega pela média."""
        mins = scores_matrix.min(axis=0)
        maxs = scores_matrix.max(axis=0)
        denom = np.where(maxs - mins > 1e-10, maxs - mins, 1.0)
        return ((scores_matrix - mins) / denom).mean(axis=1)
