"""Log the promoted eval run to MLflow, file based.

Reads one eval_runs/*.json, does not run a new eval. Phase 4 needs proof
MLflow works, not a re-run of every experiment already recorded in
docs/CHECKPOINT.md. See D16 and D9 for why MLflow exists alongside Vertex.
"""
import json
import sys
import mlflow

RUN_FILE = "eval_runs/1788453485_remote_qwen3-8b.json"


def main():
    d = json.load(open(RUN_FILE))

    # sqlite, not a folder store: recent mlflow refuses the filesystem backend outright.
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("rag-eval")

    with mlflow.start_run(run_name=d["label"]):
        mlflow.log_param("model", d["model"])
        mlflow.log_param("num_ctx", d["num_ctx"])
        mlflow.log_param("target", d["target"])
        for k, v in d["knobs"].items():
            mlflow.log_param(f"knob_{k}", v)

        mlflow.log_metric("passed", d["passed"])
        mlflow.log_metric("cases", d["cases"])
        mlflow.log_metric("pass_rate", d["passed"] / d["cases"])
        mlflow.log_metric("elapsed_s", d["elapsed_s"])
        for cat, stats in d["by_category"].items():
            mlflow.log_metric(f"cat_{cat}_pass", stats["pass"])
            mlflow.log_metric(f"cat_{cat}_n", stats["n"])

        mlflow.log_artifact(RUN_FILE)

    print(f"logged {RUN_FILE} to mlruns/, experiment 'rag-eval'")


if __name__ == "__main__":
    # self check: run file exists and has the fields this script reads
    d = json.load(open(RUN_FILE))
    for key in ("label", "model", "num_ctx", "target", "knobs", "passed", "cases", "elapsed_s", "by_category"):
        assert key in d, f"missing field: {key}"
    main()
