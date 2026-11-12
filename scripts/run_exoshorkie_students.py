"""Run students or ensembles in disjoint GPU shards, with safe pair-level resume.

Each worker owns whole students and separate result directories. Successful
pairs are skipped only when their config hash, model settings and device match.
Logs/progress live under the config's output directory; comparison runs once
all workers have finished, avoiding concurrent writes to comparison artifacts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time
import traceback

from yeastbench.config import load_config


def worker(args):
    import torch
    from yeastbench.cli import _run_pair
    from yeastbench.data.fetch import check_run_data, default_data_root

    torch.set_num_threads(2)
    torch.cuda.set_device(args.gpu)
    cfg = replace(load_config(args.config), device=f"cuda:{args.gpu}")
    digest = hashlib.sha256()
    for path in sorted(Path("src/yeastbench").rglob("*.py")):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    code_hash = digest.hexdigest()
    runs = cfg.runs[args.shard_index::args.num_shards]
    check = check_run_data(
        sorted({task for run in runs for task in run.tasks}),
        [run.model for run in runs], default_data_root(),
    )
    if not check.ready:
        raise RuntimeError(f"Missing/stale inputs: {check.missing} {check.stale}")
    progress_path = cfg.out_dir / "run_logs" / f"gpu{args.gpu}.json"
    progress = {"device": cfg.device, "config_hash": cfg.source_hash,
                "complete": [], "failed": [], "active": None}

    def save():
        progress_path.write_text(json.dumps(progress, indent=2))

    for run in runs:
        for task in run.tasks:
            pair = f"{run.model}__{task}"
            directory = cfg.out_dir / pair
            if args.resume and (directory / "summary.json").exists():
                try:
                    metadata = json.loads((directory / "run_metadata.json").read_text())
                    matches = (metadata["config_hash"] == cfg.source_hash
                               and metadata["model_config"] == run.model_config
                               and metadata.get("code_sha256") == code_hash
                               and metadata["device"] == cfg.device)
                except (OSError, ValueError, KeyError):
                    matches = False
                if matches:
                    print(f"RESUME {pair}", flush=True)
                    progress["complete"].append(pair)
                    save()
                    continue
            progress["active"] = pair
            save()
            print(f"START {pair} on {cfg.device}", flush=True)
            try:
                _run_pair(cfg, run.model, task, run.model_config)
                metadata_path = directory / "run_metadata.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["code_sha256"] = code_hash
                metadata_path.write_text(json.dumps(metadata, indent=2))
            except Exception:
                traceback.print_exc()
                progress["failed"].append(pair)
            else:
                progress["complete"].append(pair)
                print(f"COMPLETE {pair}", flush=True)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                progress["active"] = None
                save()
    return int(bool(progress["failed"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exoshorkie_students.yaml")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare-config", help="Optional config including existing comparison runs")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logs = cfg.out_dir / "run_logs"
    logs.mkdir(parents=True, exist_ok=True)
    if args.gpu is not None:
        return worker(args)
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)):
        parser.error("GPU indices must be nonempty and unique")
    processes, handles = [], []
    try:
        for shard, gpu in enumerate(gpus):
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--config", args.config, "--gpu", str(gpu),
                       "--shard-index", str(shard), "--num-shards", str(len(gpus))]
            if args.resume:
                command.append("--resume")
            handle = (logs / f"gpu{gpu}.log").open("a")
            handles.append(handle)
            processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                              env={**os.environ, "PYTHONUNBUFFERED": "1"}))
            print(f"GPU {gpu}: PID {processes[-1].pid}, log {handle.name}", flush=True)
        while any(process.poll() is None for process in processes):
            time.sleep(5)
        from yeastbench.compare import compare
        compare(load_config(args.compare_config) if args.compare_config else cfg)
        return int(any(process.returncode for process in processes))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
