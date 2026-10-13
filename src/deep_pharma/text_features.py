from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_FIELDS = (
    "Mechanism",
    "Pharmacodynamics",
    "Metabolism",
    "Half_Life",
    "Absorption",
    "Toxicity",
    "Protein_Binding",
    "Elimination_Route",
    "Indication",
    "Warning",
    "CYP450_Enzymes",
)


def normalize_biomedical_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value))
    return re.sub(r"\s+", " ", text).strip()


@cache
def _cached_molecule_document(values: tuple[str, ...], include_warning_toxicity: bool) -> str:
    sections = []
    for field, raw_value in zip(TEXT_FIELDS, values, strict=True):
        if not include_warning_toxicity and field in {"Warning", "Toxicity"}:
            continue
        value = normalize_biomedical_text(raw_value)
        sections.append(f"[{field.upper()}] {value}")
        sections.append(f"[{field.upper()}_MISSING] {int(not value)}")
    return "\n".join(sections)


def molecule_documents(
    frame: pd.DataFrame, suffix: str, *, include_warning_toxicity: bool
) -> list[str]:
    documents = []
    for _, row in frame.iterrows():
        values = tuple(
            "" if pd.isna(row[f"{field}_{suffix}"]) else str(row[f"{field}_{suffix}"])
            for field in TEXT_FIELDS
        )
        documents.append(_cached_molecule_document(values, include_warning_toxicity))
    return documents


def _absolute_sparse(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    result = matrix.tocsr(copy=True)
    result.data = np.abs(result.data)
    return result


@dataclass
class FoldTextEncoder:
    n_components: int = 64
    word_max_features: int = 20000
    char_max_features: int = 8000
    min_df: int = 3
    random_state: int = 2026
    include_warning_toxicity: bool = True

    def __post_init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_features=self.word_max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=self.min_df,
            max_features=self.char_max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)

    def _documents(self, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        return (
            molecule_documents(frame, "A", include_warning_toxicity=self.include_warning_toxicity),
            molecule_documents(frame, "B", include_warning_toxicity=self.include_warning_toxicity),
        )

    def _vectorize_documents(self, documents: list[str]) -> sparse.csr_matrix:
        unique_documents = list(dict.fromkeys(documents))
        document_to_index = {document: index for index, document in enumerate(unique_documents)}
        encoded_unique = sparse.hstack(
            [
                self.word_vectorizer.transform(unique_documents),
                self.char_vectorizer.transform(unique_documents),
            ],
            format="csr",
        )
        row_indices = [document_to_index[document] for document in documents]
        return encoded_unique[row_indices]

    def _pair_matrix(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        documents_a, documents_b = self._documents(frame)
        encoded_a = self._vectorize_documents(documents_a)
        encoded_b = self._vectorize_documents(documents_b)
        return sparse.hstack(
            [
                encoded_a + encoded_b,
                _absolute_sparse(encoded_a - encoded_b),
                encoded_a.multiply(encoded_b),
            ],
            format="csr",
        )

    def fit(self, frame: pd.DataFrame) -> FoldTextEncoder:
        documents_a, documents_b = self._documents(frame)
        # Drug descriptions repeat across many pairs. Deduplicating exact documents
        # avoids overweighting popular molecules and keeps the fold-local fit compact.
        corpus = list(dict.fromkeys(documents_a + documents_b))
        self.word_vectorizer.fit(corpus)
        self.char_vectorizer.fit(corpus)
        pair_matrix = self._pair_matrix(frame)
        available_components = min(pair_matrix.shape[0] - 1, pair_matrix.shape[1] - 1)
        self.svd.set_params(n_components=min(self.n_components, available_components))
        self.svd.fit(pair_matrix)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.svd.transform(self._pair_matrix(frame)).astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)
