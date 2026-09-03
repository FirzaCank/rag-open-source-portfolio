# RAG Open Source Portfolio

A portfolio chat assistant that answers questions about me from my own data, rebuilt on fully
open source models and served from a GPU that scales to zero.

### Live demo

**https://portfolio-web-121020284283.asia-southeast1.run.app**

The chat widget is bottom right. First question after an idle period takes about 92 seconds,
because the GPU service scales to zero and has to cold start. Every question after that answers
in about 3 seconds.

| | |
| :--- | :--- |
| Demo, open source models on Cloud Run GPU | https://portfolio-web-121020284283.asia-southeast1.run.app |
| API, Swagger UI | `https://rag-api-121020284283.asia-southeast1.run.app/docs` |
| The production site this was ported from, runs Gemini | https://firzacank.vercel.app |

The Vercel site is the comparison anchor, not the deliverable. It scores 30 of 41 on the same
frozen test set. This repository replaces every closed model in it with an open one and measures
what that costs.

Final project, AI Super Class.

## What is inside

| Layer | Choice |
| :--- | :--- |
| Generation | `qwen3:8b` served by Ollama, promoted on 36 of 41 at two seeds with an identical failure list |
| Embedding | multilingual e5 base, 768 dim, fastembed on ONNX Runtime |
| Vector search | numpy cosine over 76 chunks, in memory. No vector database |
| Tools | 6 function tools, an agentic loop capped at 2 rounds, then one pass with the tools withdrawn |
| Guards | three deterministic filters in code, on the tool call path, not in the prompt |
| API | FastAPI, Swagger UI at `/docs`, chat is a raw text stream |
| Web UI | the existing Next.js site, deployed as a second Cloud Run service, chat proxied server side |
| Serving | two Cloud Run services in `asia-southeast1`, both scale to zero |

Every row was chosen against alternatives that lost on measured grounds. The reasoning, the
numbers, and the rejected options live in the project decision log, kept outside this repository.

## Status

Phases 0 through 3d are done. Both services are deployed and scale to zero. Phase 4 is in
progress: data and experiment versioning are done, canary rollout is next.

| Measured | Value |
| :--- | :--- |
| Corpus | 39 documents, 76 chunks, 22 sources, 721 KB index |
| Retrieval, 66 hand-written queries | Hit Rate@1 68.2, Hit Rate@8 86.4, MRR@8 0.7497 |
| Answer quality, 41 frozen cases | **36 of 41**, against 30 for the Gemini version |
| Hallucination rate | 1 of 41, hand classified |
| Latency, warm, measured on the frozen 7B baseline | first token 0.88 s p50, full answer 2.57 s p50 |
| Cold start, from zero | 92 s, the designed cost of scaling to zero |

## Versioning

| What | Where |
| :--- | :--- |
| Corpus and index | DVC, snapshot at `data/versions/v1-variant-a/`, GCS remote |
| Experiment runs | MLflow, sqlite backend, pushed to GCS |
| Promoted config | Vertex AI Model Registry, `rag-portfolio-qwen3-v13` |
| Container images | Artifact Registry, tagged per Cloud Build run |

The RAG config, not the model, is what gets registered. `qwen3:8b` is used as is, no fine-tune.
What is versioned is the prompt variant, the guard set, and the eval score that came with them.

Three of the guards were added because the prompt could not do the job. Tool names leaked in 8 of
8 runs while the prompt forbade it, so a filter now cuts the stream at the first tool name. The
model sent a message using an email address it invented, so the send tool rejects an address that
appears in no visitor turn. Both were measured before and after, with zero regressions.

The 41 cases are frozen in content and in regex, so a correct answer phrased differently still
fails. That is deliberate: loosening a regex to make one model pass destroys the comparison the
set exists for. Failures are classified by hand instead.

## Running it

Needs Python 3.10, the version every number above was produced on, and
[Ollama](https://ollama.com) with the model pulled.

```bash
pip install -r requirements.txt
ollama pull qwen3:8b
```

The index is committed, so retrieval works without rebuilding it. Rebuild only after changing the
corpus, the chunking, or the embedding model:

```bash
python scripts/build_index.py
```

Every module with non-trivial logic carries its own self check, which is the fastest way to see
whether the environment is correct:

```bash
python -m rag.retriever    # index loads, 76 chunks at 768 dim
python -m rag.llm          # end to end answer, needs Ollama running
```

Serve the API, Swagger UI at `/docs`:

```bash
uvicorn app:api --port 8080
```

Run the evaluation. It refuses to score a run where any case errored, and it records which model
the server actually served rather than which one the local environment names:

```bash
python scripts/eval_chat.py --label "local run"
python scripts/compare_runs.py eval_runs/<before>.json eval_runs/<after>.json
```
