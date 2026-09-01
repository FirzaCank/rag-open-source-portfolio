# Throwaway image for the Phase 0 GPU smoke test. Not the production image.
# The real one, Ollama plus FastAPI, is written in Phase 3.

# A CUDA base image is what makes nvidia-smi resolvable once Cloud Run mounts the driver.
# Tag is pinned rather than :latest so a rebuild next week tests the same thing.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# python3-minimal only. The smoke test imports nothing outside the standard library, so pip,
# venv, and the rest would be pure image weight and pure build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-minimal \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY main.py .

# No CMD shell form, so Python is PID 1 and receives Cloud Run's SIGTERM directly.
CMD ["python3", "main.py"]
