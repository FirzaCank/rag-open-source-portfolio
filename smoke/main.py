"""Throwaway GPU smoke test for Cloud Run.

Phase 0 exists to answer one question before any RAG code is written: does this GCP project
actually give us an L4 on Cloud Run? Quota, region, billing, and the zonal redundancy flag can
each block that, and finding out in the final week is the failure this file prevents. See D18.

It serves exactly one endpoint, `/`, which shells out to `nvidia-smi` and returns whatever it
says. No dependencies beyond the Python standard library, so nothing here can fail for a reason
unrelated to the GPU.

Deleted at the end of Phase 0. Nothing else in the repo imports it.
"""

import http.server
import os
import subprocess

# Cloud Run injects PORT and expects the container to listen on it. Anything else and the
# revision is marked unhealthy no matter how healthy the GPU is.
PORT = int(os.environ.get("PORT", "8080"))


def gpu_report() -> str:
    """Return nvidia-smi output, or the reason it could not run.

    Returning the error as text instead of raising is deliberate. A crash here would show up in
    Cloud Run as a dead container, which looks identical to a hundred other startup failures. A
    200 response carrying the error message tells us which of the two problems we have.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return "nvidia-smi not found on PATH. The GPU driver was not mounted into the container."
    except subprocess.TimeoutExpired:
        return "nvidia-smi timed out after 30s."

    if out.returncode != 0:
        return f"nvidia-smi exited {out.returncode}\n\nstderr:\n{out.stderr}"
    return out.stdout


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = gpu_report().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Default BaseHTTPRequestHandler logging writes to stderr, which Cloud Logging reads as
        # ERROR severity. Send it to stdout instead so a normal request is not an alert.
        print(fmt % args, flush=True)


if __name__ == "__main__":
    # Printed once at startup so the answer is in the logs even if the URL is never opened.
    print("startup gpu check:", flush=True)
    print(gpu_report(), flush=True)
    print(f"listening on 0.0.0.0:{PORT}", flush=True)
    http.server.HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
