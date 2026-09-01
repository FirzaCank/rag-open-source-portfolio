# Throwaway image for the Phase 0 GPU smoke test. The production one is written in Phase 3.

# A CUDA base makes nvidia-smi resolvable once Cloud Run mounts the driver. Pinned, not :latest.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# python3, not python3-minimal. v1 died on `import http.server`: python3-minimal omits most of the
# standard library, and libpython3.10-stdlib was only a recommended package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

# Fail the build, not the deploy. A missing import surfaces in seconds instead of as a Cloud Run
# health check timeout that says nothing about the cause.
RUN python3 -c "import http.server, subprocess, os; print('stdlib check ok')"

WORKDIR /app
COPY main.py .

# No CMD shell form, so Python is PID 1 and receives Cloud Run's SIGTERM directly.
CMD ["python3", "main.py"]
