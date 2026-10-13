from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer

from .chemistry_features import build_chemical_pair_features
from .pk_features import build_pk_pair_features
from .text_features import FoldTextEncoder


@dataclass
class MultimodalFeatureBuilder:
    text_components: int = 64
    morgan_components: int = 64
    word_max_features: int = 20000
    char_max_features: int = 8000
    min_df: int = 3
    include_warning_toxicity: bool = True
    random_state: int = 2026

    def __post_init__(self) -> None:
        self.text_encoder = FoldTextEncoder(
            n_components=self.text_components,
            word_max_features=self.word_max_features,
            char_max_features=self.char_max_features,
            min_df=self.min_df,
            include_warning_toxicity=self.include_warning_toxicity,
            random_state=self.random_state,
        )
        self.morgan_svd = TruncatedSVD(
            n_components=self.morgan_components, random_state=self.random_state
        )
        self.numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)

    @staticmethod
    def _numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
        _, descriptors = build_chemical_pair_features(frame)
        pk_features = build_pk_pair_features(frame)
        return pd.concat([descriptors, pk_features], axis=1).replace([np.inf, -np.inf], np.nan)

    def fit(self, frame: pd.DataFrame) -> MultimodalFeatureBuilder:
        morgan, _ = build_chemical_pair_features(frame)
        available_components = min(morgan.shape[0] - 1, morgan.shape[1] - 1)
        self.morgan_svd.set_params(n_components=min(self.morgan_components, available_components))
        self.morgan_svd.fit(morgan)
        self.text_encoder.fit(frame)
        self.numeric_imputer.fit(self._numeric_features(frame))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        morgan, _ = build_chemical_pair_features(frame)
        morgan_reduced = self.morgan_svd.transform(morgan).astype(np.float32)
        text_reduced = self.text_encoder.transform(frame)
        numeric = self.numeric_imputer.transform(self._numeric_features(frame)).astype(np.float32)
        features = np.hstack([morgan_reduced, text_reduced, numeric]).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Le preprocessing a produit des valeurs non finies.")
        return features

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)

    @property
    def family_dimensions(self) -> dict[str, int]:
        return {
            "morgan_svd": int(self.morgan_svd.n_components),
            "text_tfidf_svd": int(self.text_encoder.svd.n_components),
            "rdkit_and_pk": int(self.numeric_imputer.statistics_.shape[0]),
        }
