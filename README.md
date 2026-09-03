# RAG Open Source Portfolio

Portfolio chat assistant for [firzacank.vercel.app](https://firzacank.vercel.app), rebuilt on
fully open source models and deployed to Cloud Run GPU. The production version of this assistant
runs on Gemini. This repository replaces every closed model in it with an open one and measures
what that costs.

Final project, AI Super Class.

## What is inside

| Layer | Choice |
| :--- | :--- |
| Generation | Qwen2.5-7B-Instruct, Q4, served by Ollama |
| Embedding | multilingual e5 base, 768 dim, fastembed on ONNX Runtime |
| Vector search | numpy cosine over 76 chunks, in memory |
| API | FastAPI, Swagger UI at `/docs` |
| Web UI | the existing Next.js site, deployed as a second Cloud Run service |
| Serving | two Cloud Run services in `asia-southeast1`, both scale to zero |

Every row above was chosen against alternatives that lost on measured grounds, not on taste.
The reasoning, the numbers, and the rejected options are written up in the project's decision log,
which is kept outside this repository. The architecture document produced in Phase 6 carries the
conclusions.

## Status

Phases 0 through 3d are done. Both services are deployed on Cloud Run and scale to zero. The
assignment is submittable from Phase 3 onward, and the remaining phases are measurement, not
plumbing.

| Measured | Value |
| :--- | :--- |
| Corpus | 39 documents, 76 chunks, 22 sources |
| Retrieval, 66 hand-written queries | Hit Rate@1 68.2, Hit Rate@8 86.4, MRR@8 0.7497 |
| Answer quality, 41 frozen cases | 29 of 41 |
| Hallucination rate | 1 of 41, hand classified |
| Latency, warm | first token 0.9 s median, full answer 2.6 s median |
| Cold start, from zero | 92 s, the designed cost of scaling to zero |

The 41 cases are frozen in content and in regex, so a correct answer phrased differently still
fails. That is deliberate: loosening a regex to make one model pass destroys the comparison the
set exists for. Failures are classified by hand instead.

Service URLs are deliberately not published here. Waking the GPU costs money, and a URL in a
public README is an invitation to do it.

## Running it

Needs Python 3.10, the version every number above was produced on, and
[Ollama](https://ollama.com) with the model pulled.

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct-q4_K_M
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
