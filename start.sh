#!/bin/sh
# Container entrypoint: Ollama first, model warmed, then the API binds the port.
#
# Order matters. Cloud Run marks the container ready as soon as something listens on $PORT, so if
# uvicorn bound first, the service would report healthy while the model was still loading and the
# first visitor would pay for it. The widget aborts after 60 seconds, so that wait is not free.
# See D52.
set -e

ollama serve &
OLLAMA_PID=$!

until curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
  sleep 0.5
done
echo '{"evt":"boot","step":"ollama_up"}'

# Pull the weights into VRAM now. The model is already on disk, baked at build time, see D44.
curl -sf http://127.0.0.1:11434/api/generate \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
  > /dev/null 2>&1 || echo '{"evt":"boot","step":"warmup_failed"}'
echo '{"evt":"boot","step":"model_warm"}'

# exec so uvicorn becomes PID 1's child and receives Cloud Run's shutdown signal directly.
exec uvicorn app:api --host 0.0.0.0 --port "${PORT:-8080}"
