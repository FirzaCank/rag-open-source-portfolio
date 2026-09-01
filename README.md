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

Phase 0 of 12: repository skeleton and a Cloud Run GPU smoke test.

## Running it

Not yet runnable. This section gets written in Phase 3, once the container builds.
