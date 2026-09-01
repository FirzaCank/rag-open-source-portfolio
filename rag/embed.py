"""Embeddings via fastembed on ONNX Runtime. multilingual e5 base, 768 dimensions.

Replaces `lib/rag/embed.ts`, which called the Gemini embedding API over HTTP. Same 768 dimensions,
so the index has the same shape, which is exactly why the guard in the retriever exists. See D27.

**e5 base is not in fastembed's built in model list.** Only `intfloat/multilingual-e5-large` ships
by default, 1024 dimensions and 2.24 GB. Base is registered here by hand against the official
`intfloat/multilingual-e5-base` repository, which publishes `onnx/model.onnx` itself, so no third
party mirror is involved and the model card in Phase 5c can point at the upstream author.

Staying on base rather than switching to the built in large is deliberate. D27 chose base, Phase 5c
fine-tunes base, and swapping now would change two things at once with no way to attribute a
difference to either. See D47.

The `query:` and `passage:` prefixes are mandatory, not decoration. e5 was trained with them, and
dropping them silently changes the task the model thinks it is doing. See D27.

No torch. This runs on ONNX Runtime, which fastembed pulls in. See D48.
"""

import os

import numpy as np
from fastembed.common.model_description import (
    DenseModelDescription,
    ModelSource,
    PoolingType,
)
from fastembed.text.custom_text_embedding import CustomTextEmbedding
from fastembed import TextEmbedding

MODEL_NAME = "intfloat/multilingual-e5-base"
DIM = 768

# Recorded because the file changes the vectors while MODEL_NAME and DIM stay the same. The same
# repo offers an int8 file at 279 MB against 1110 MB, a Phase 3 cold start lever. See D27.
MODEL_FILE = "onnx/model.onnx"

# fastembed caches to temp, which macOS and Cloud Run clear. 1.1 GB lands in git ignored `models/`.
_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("FASTEMBED_CACHE", os.path.join(_HERE, "..", "models", "fastembed"))

_model: TextEmbedding | None = None
_registered = False


def _register() -> None:
    """Teach fastembed about e5 base. Idempotent, because import order is not ours to control.

    `pooling=MEAN` and `normalization=True` are the model's own recipe: e5 mean pools the token
    embeddings and L2 normalises the result. Getting either wrong produces vectors that still have
    768 dimensions and still load without error, and retrieval quality quietly drops.
    """
    global _registered
    if _registered:
        return
    # list_supported_models() returns plain dicts, not model description objects.
    if any(m["model"] == MODEL_NAME for m in TextEmbedding.list_supported_models()):
        _registered = True
        return

    CustomTextEmbedding.add_model(
        DenseModelDescription(
            model=MODEL_NAME,
            dim=DIM,
            description="multilingual e5 base, 768 dim, ONNX from the official intfloat repository",
            license="mit",
            size_in_GB=1.11,
            sources=ModelSource(hf=MODEL_NAME),
            model_file=MODEL_FILE,
        ),
        pooling=PoolingType.MEAN,
        normalization=True,
    )
    _registered = True


def get_model() -> TextEmbedding:
    """Load the model once, on first use.

    Lazy on purpose. Loading at import time would put a 1.1 GB read into the cold start of every
    process that touches this module, including scripts that only need the corpus.
    """
    global _model
    if _model is None:
        _register()
        os.makedirs(CACHE_DIR, exist_ok=True)
        _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    return _model


def _embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(get_model().embed(texts)), dtype=np.float32)
    if vectors.shape[1] != DIM:
        raise ValueError(f"expected {DIM} dimensions, model returned {vectors.shape[1]}")
    return vectors


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed corpus chunks. Shape `(len(texts), 768)`."""
    return _embed([f"passage: {t}" for t in texts])


def embed_query(text: str) -> np.ndarray:
    """Embed one visitor question. Shape `(768,)`."""
    return _embed([f"query: {text}"])[0]


if __name__ == "__main__":
    # First run downloads about 1.1 GB into models/fastembed and takes a few minutes.
    print(f"model {MODEL_NAME}, file {MODEL_FILE}, cache {os.path.abspath(CACHE_DIR)}")

    passages = [
        "Firza worked as a Data Engineer at Hypefast, building data pipelines.",
        "The Telco Customer Churn Prediction project used Streamlit and scikit-learn.",
    ]
    P = embed_passages(passages)
    assert P.shape == (2, DIM), P.shape
    assert P.dtype == np.float32, P.dtype

    # Unit length is what lets the retriever use a dot product as cosine. If this fails, pooling or
    # normalisation is wrong, and every score downstream is wrong with it.
    norms = np.linalg.norm(P, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), norms

    q = embed_query("Where did Firza work as a data engineer?")
    assert q.shape == (DIM,), q.shape
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-3

    # The right passage must win. Smallest end to end statement that retrieval works at all: a
    # question about the employer beats the churn project passage.
    scores = P @ q
    assert scores[0] > scores[1], f"wrong passage won: {scores}"

    # Embedding the same question as a passage must give a different vector. If it does not, the
    # prefixes are ignored and D27 is violated silently.
    as_passage = embed_passages(["Where did Firza work as a data engineer?"])[0]
    drift = float(np.linalg.norm(q - as_passage))
    assert drift > 1e-3, f"query and passage prefixes produced the same vector, drift {drift}"

    print(f"scores: employment {scores[0]:.4f}, churn project {scores[1]:.4f}")
    print(f"prefix drift between query and passage encoding: {drift:.4f}")
    print(f"vector norms: {norms.round(6)}")
    print("all checks passed")
