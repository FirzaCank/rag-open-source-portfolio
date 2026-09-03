"""Build data/embeddings.json: chunk the corpus, embed every chunk, write the index.

Replaces `scripts/build-embeddings.ts`, which called the Gemini API. Rerun this after any change to
the corpus, to the chunking, or to the embedding model. Nothing else in the project writes the
index.

**The metadata block is the point, not the vectors.** The Gemini index was a bare JSON list of
`{source, text, vector}`. This one is an object carrying `model`, `dim`, and `model_file` alongside
the chunks, so the retriever can refuse a file it does not understand. multilingual e5 base is 768
dimensions, the same shape as the old Gemini index, so a mismatched file loads without a single
error and turns retrieval into noise. See D27.

`model_file` is there because quantisation changes the vectors while leaving `model` and `dim`
untouched. An index built on `onnx/model.onnx` and served by the 279 MB int8 file would pass a
check on name and dimension alone.

Run: python scripts/build_index.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag.chunk import OVERLAP_CHARS, TARGET_CHARS, VARIANT, chunk_docs
from rag.embed import DIM, MODEL_FILE, MODEL_NAME, embed_passages
from rag.sources import get_docs

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "embeddings.json")


def build() -> dict:
    docs = get_docs()
    chunks = chunk_docs(docs)
    print(f"{len(docs)} documents, {len(chunks)} chunks, "
          f"{len({c['source'] for c in chunks})} sources")

    print(f"embedding with {MODEL_NAME} ({MODEL_FILE}), this takes a minute on CPU")
    vectors = embed_passages([c["text"] for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(f"got {len(vectors)} vectors for {len(chunks)} chunks")

    return {
        "model": MODEL_NAME,
        "dim": DIM,
        "model_file": MODEL_FILE,
        # Read off CHUNK_VARIANT, not hardcoded: a B index once recorded itself as A. See D89.
        "chunking": {"variant": VARIANT, "target_chars": TARGET_CHARS, "overlap_chars": OVERLAP_CHARS},
        "chunks": [
            {"source": c["source"], "text": c["text"], "vector": [round(float(x), 7) for x in v]}
            for c, v in zip(chunks, vectors)
        ],
    }


if __name__ == "__main__":
    index = build()

    # Rounding to 7 decimals halves the file and moves a norm by 1e-7. Checked, not assumed.
    import numpy as np

    V = np.array([c["vector"] for c in index["chunks"]], dtype=np.float32)
    norms = np.linalg.norm(V, axis=1)
    assert V.shape == (len(index["chunks"]), DIM), V.shape
    assert np.allclose(norms, 1.0, atol=1e-4), f"norms drifted: min {norms.min()}, max {norms.max()}"

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)

    kb = os.path.getsize(OUT_PATH) / 1024
    print(f"wrote {len(index['chunks'])} chunks to data/embeddings.json ({kb:.0f} KB)")
    print(f"model {index['model']}, dim {index['dim']}, file {index['model_file']}")
    print(f"norms: min {norms.min():.6f}, max {norms.max():.6f}")
