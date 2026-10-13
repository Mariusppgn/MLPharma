"""Per-builder exact-document transform memo; never shared between fitted folds."""

from scipy import sparse


def cache_fitted_text_transforms(builder):
    """Install before first fit; do not refit this builder with another vocabulary."""
    document_cache = {}
    matrix_cache = {}
    original_vectorize = builder.text_encoder._vectorize_documents

    def cached_vectorize(documents):
        key = tuple(documents)
        if key not in matrix_cache:
            unseen = list(dict.fromkeys(doc for doc in documents if doc not in document_cache))
            if unseen:
                encoded = original_vectorize(unseen).tocsr()
                for row, document in enumerate(unseen):
                    document_cache[document] = encoded[row]
            matrix_cache[key] = sparse.vstack(
                [document_cache[document] for document in documents], format="csr"
            )
        return matrix_cache[key]

    builder.text_encoder._vectorize_documents = cached_vectorize
    return builder


def clear_fitted_text_transform_cache(builder):
    """Restore the class method before pickling or refitting the feature builder."""
    builder.text_encoder.__dict__.pop("_vectorize_documents", None)
    return builder
