"""Phase 5c: fine-tune multilingual-e5-base on 163 portfolio pairs, export ONNX.

Runs on a Colab T4. Paste each `# %%` block into its own cell, or run the file as a script in a
Colab that has the repo cloned. Nothing here is a serving dependency: `torch` and
`sentence-transformers` live in Colab only, and production loads the exported ONNX through
fastembed. See D46 and D48.

**Why this exists.** Four retrieval arms were measured and all four rejected: `top_k=16` (33 of 41
against 36), MMR (cannot promote a candidate at 15 of 16), paragraph chunking (HR@1 68.2 to 59.1),
and the e5 prefixes were already correct. What is left is the embedding model itself. `fact-education`
fails because "Where did Firza study?" scores 0.7298 against the chunk holding `Education: Bachelor's
Degree ... ITB` and loses to `Career Timeline` at 0.8386, which contains no education text at all.
That is a vocabulary gap, and a fine-tune is the lever that closes it. See D89 and D90.

**The prefixes are the thing most likely to go wrong.** e5 was trained with `query:` on questions
and `passage:` on documents, and `rag/embed.py` applies them at inference. Training without them, or
with them swapped, produces a model that scores well in Colab and degrades in production with no
error anywhere. Cell 3 asserts both are present before a single step runs.

**Kill criteria, fixed before training starts.** Any one of these rejects the arm, and a fired
criterion gets reported rather than tuned around:
  - HR@1 over the 66 query retrieval set below 68.2
  - HR@1 over the 12 Indonesian queries below 66.7
  - golden set below 36 of 41
Baseline to beat: all 68.2 / 81.8 / 86.4 / MRR 0.7497, english 68.5, indonesian 66.7. See D47.
"""

# %% Cell 1: install
# sentence-transformers pulls torch, which is why this runs in Colab and never in the image.
# !pip install -q sentence-transformers==3.0.1 onnx onnxruntime

# %% Cell 2: clone the repo for the pairs and the eval set
# !git clone https://github.com/FirzaCank/rag-open-source-portfolio.git
# %cd rag-open-source-portfolio

# %% Cell 3: load pairs and verify the prefixes
import json
from pathlib import Path

PAIRS_PATH = Path("data/train_pairs.json")
BASE_MODEL = "intfloat/multilingual-e5-base"
OUT_DIR = Path("finetuned-e5-base")

pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))["pairs"]
print(f"{len(pairs)} pairs, {len({p['query'] for p in pairs})} distinct queries")

# e5 needs `query:` on questions and `passage:` on documents. See the docstring.
train = [(f"query: {p['query']}", f"passage: {p['text']}") for p in pairs]
assert all(a.startswith("query: ") and b.startswith("passage: ") for a, b in train)
assert len({a for a, _ in train}) == len(train), "a query appears twice, positives are contradictory"
print(f"prefixes verified on {len(train)} pairs")

# %% Cell 4: train
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

model = SentenceTransformer(BASE_MODEL)

# In-batch negatives, no hand written hard negatives: nothing measured shows they are needed.
loader = DataLoader([InputExample(texts=[q, t]) for q, t in train], shuffle=True, batch_size=16)
loss = losses.MultipleNegativesRankingLoss(model)

# 163 pairs is small, so 3 epochs and a low LR. More epochs on this little data overfits.
model.fit(
    train_objectives=[(loader, loss)],
    epochs=3,
    warmup_steps=10,
    optimizer_params={"lr": 2e-5},
    show_progress_bar=True,
)
model.save(str(OUT_DIR))
print(f"saved to {OUT_DIR}")

# %% Cell 5: does it actually move the failing case?
import numpy as np

def score(q, t):
    v = model.encode([f"query: {q}", f"passage: {t}"], normalize_embeddings=True)
    return float(v[0] @ v[1])

edu = next(p["text"] for p in pairs if p["template"] == "label:Education")
timeline = next(p["text"] for p in pairs if p["template"] == "timeline")
q = "Where did Firza study?"

# Baseline: education 0.7298 loses to Career Timeline 0.8386. The gap has to invert.
print(f"education chunk : {score(q, edu):.4f}  (baseline 0.7298)")
print(f"career timeline : {score(q, timeline):.4f}  (baseline 0.8386)")
print("inverted" if score(q, edu) > score(q, timeline) else "STILL LOSING, arm is likely dead")

# %% Cell 6: export ONNX, since production never loads torch
import torch

tok = model.tokenizer
transformer = model[0].auto_model
transformer.eval()

sample = tok("query: test", return_tensors="pt", padding=True, truncation=True, max_length=512)
OUT_ONNX = OUT_DIR / "onnx"
OUT_ONNX.mkdir(parents=True, exist_ok=True)

# Dynamic axes on batch and sequence, or the exported graph only accepts one input shape.
torch.onnx.export(
    transformer,
    (sample["input_ids"], sample["attention_mask"]),
    str(OUT_ONNX / "model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    },
    opset_version=14,
)
tok.save_pretrained(str(OUT_DIR))
print(f"exported {OUT_ONNX / 'model.onnx'}")

# %% Cell 7: the ONNX must match the torch model, or production serves a different model
import onnxruntime as ort

sess = ort.InferenceSession(str(OUT_ONNX / "model.onnx"), providers=["CPUExecutionProvider"])
probe = "query: Where did Firza study?"
enc = tok(probe, return_tensors="np", padding=True, truncation=True, max_length=512)

onnx_out = sess.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})[0]
with torch.no_grad():
    torch_out = transformer(**tok(probe, return_tensors="pt", padding=True,
                                  truncation=True, max_length=512)).last_hidden_state.numpy()

# Mean pooling is what e5 and fastembed both use, so compare the pooled vector, not raw states.
def pooled(h, mask):
    m = mask[..., None].astype("float32")
    return (h * m).sum(1) / m.sum(1)

a = pooled(onnx_out, enc["attention_mask"])
b = pooled(torch_out, enc["attention_mask"])
a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
cos = float((a * b).sum())
print(f"onnx vs torch cosine: {cos:.6f}")
assert cos > 0.9999, f"ONNX diverged from torch at {cos}, do not ship this export"

# %% Cell 8: zip for download, then rebuild the index locally and run both evals
# !zip -qr finetuned-e5-base.zip finetuned-e5-base && ls -lh finetuned-e5-base.zip
# from google.colab import files; files.download("finetuned-e5-base.zip")
