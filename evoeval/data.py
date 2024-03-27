# largely adopted from EvalPlus
import gzip
import hashlib
import json
import os
from typing import Dict, Iterable

import tempdir
import wget
from appdirs import user_cache_dir

CACHE_DIR = user_cache_dir("evoeval")


EVOEVAL_VERSION = "v0.1.0"
EVOEVAL_OVERRIDE_PATH = os.environ.get("EVOEVAL_OVERRIDE_PATH", None)


def write_jsonl(
    filename: str, data: Iterable[Dict], append: bool = False, drop_builtin: bool = True
):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    if drop_builtin:
                        x = {k: v for k, v in x.items() if not k.startswith("_")}
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                if drop_builtin:
                    x = {k: v for k, v in x.items() if not k.startswith("_")}
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def make_cache(gzip_url, cache_path, dataset_name):
    # Check if human eval file exists in CACHE_DIR
    if not os.path.exists(cache_path):
        # Install HumanEval dataset and parse as jsonl
        print(f"Downloading dataset from {gzip_url}")
        with tempdir.TempDir() as tmpdir:
            # TODO need to test this.
            evoeval_gz_path = os.path.join(tmpdir, f"{dataset_name}-data.jsonl.gz")
            wget.download(gzip_url, evoeval_gz_path)

            with gzip.open(evoeval_gz_path, "rb") as f:
                evoeval = f.read().decode("utf-8")

        # create CACHE_DIR if not exists
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        # Write the original human eval file to CACHE_DIR
        with open(cache_path, "w") as f:
            f.write(evoeval)


def get_dataset_metadata(name: str, version: str):
    assert name in [
        "EvoEval_difficult",
        "EvoEval_creative",
        "EvoEval_subtle",
        "EvoEval_combine",
        "EvoEval_tool_use",
        "EvoEval_verbose",
        "EvoEval_concise",
    ], f"Unknown/unsupported dataset: {name}"
    url = f"https://github.com/evo-eval/evoeval_release/releases/download/{version}/{name}.jsonl.gz"
    cache_path = os.path.join(CACHE_DIR, f"{name}-{version}.jsonl")
    return url, cache_path


def _ready_evo_eval_path(dataset_name: str) -> str:
    if EVOEVAL_OVERRIDE_PATH is not None:
        # create CACHE_DIR if not exists
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        return f"{EVOEVAL_OVERRIDE_PATH}/{dataset_name}.jsonl"

    url, cache_path = get_dataset_metadata(dataset_name, EVOEVAL_VERSION)
    make_cache(url, cache_path, dataset_name)

    return cache_path


def get_evo_eval_plus_hash(dataset_name: str) -> str:
    evoeval_path = _ready_evo_eval_path(dataset_name)
    with open(evoeval_path, "rb") as f:
        evoeval = f.read()
    return hashlib.md5(evoeval).hexdigest()


def get_evo_eval(dataset_name: str):
    evoeval_path = _ready_evo_eval_path(dataset_name)
    with open(evoeval_path, "r") as f:
        data = {json.loads(task)["task_id"]: json.loads(task) for task in f.readlines()}

    return data
