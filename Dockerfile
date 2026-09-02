# One image, Ollama plus FastAPI. Pinned to the same Ollama version as the laptop, 0.33.0, so a
# behaviour difference between local and Cloud Run cannot be blamed on the runtime.
FROM ollama/ollama:0.33.0

# PYTHONUNBUFFERED is not a style choice. Python buffers stdout when it is not a tty, and on Cloud
# Run that means request logs are lost when the container dies. Logs are the only observability
# there, so this is a requirement.
ENV PYTHONUNBUFFERED=1 \
    MODEL=qwen2.5:7b-instruct-q4_K_M \
    OLLAMA_KEEP_ALIVE=-1 \
    FASTEMBED_CACHE=/app/models/fastembed \
    PORT=8080

# The base image ships Ollama and the CUDA runtime, not Python. curl is used by the healthcheck
# loop in start.sh and by the model pull below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv curl \
    && rm -rf /var/lib/apt/lists/*

# A venv rather than pip into the system interpreter, which newer Ubuntu refuses under PEP 668.
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"
RUN python3 -m venv "$VIRTUAL_ENV"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights into the image. Never pulled at startup, see D44. This runs before the code is
# copied so that editing a Python file does not invalidate a 4.7 GB layer.
# `ollama list` runs while the daemon is still up. An earlier version killed it first and the
# build failed on "could not connect to ollama server" after a successful 4.7 GB pull.
RUN sh -c 'set -e; \
    ollama serve & pid=$!; \
    until curl -sf http://127.0.0.1:11434/api/tags > /dev/null; do sleep 1; done; \
    ollama pull "$MODEL"; \
    ollama list; \
    kill $pid || true' \
    && test -d /root/.ollama/models/manifests/registry.ollama.ai/library/qwen2.5 \
    && du -sh /root/.ollama/models

COPY rag/ ./rag/
COPY data/ ./data/
COPY app.py start.sh ./

# The embedding model is 1.0 GB and fastembed downloads it on first use. Unbaked, the first visitor
# would wait for a HuggingFace download inside Cloud Run. Same reasoning as the weights above.
RUN python -c "from rag.embed import get_model; get_model(); print('embedder cached')"

# Fail the build, not the deploy, if the index and the embedding model disagree. A mismatched index
# loads silently and turns retrieval into noise, see D27.
RUN python -c "from rag.retriever import _load_index; s, t, v = _load_index(); print(f'index ok: {len(t)} chunks, {len(set(s))} sources, dim {v.shape[1]}')"

# The base image's entrypoint is the ollama binary. This service starts both processes.
ENTRYPOINT []
CMD ["./start.sh"]
