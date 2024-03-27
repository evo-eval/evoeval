# Adopted from https://github.com/evalplus/evalplus
import argparse
import contextlib
import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
from warnings import warn

import numpy as np
from evalplus.data import get_human_eval_plus
from evalplus.data.utils import load_solutions
from evalplus.gen.util import trusted_exec
from termcolor import cprint
from tqdm import tqdm

from evoeval.data import CACHE_DIR, get_evo_eval, get_evo_eval_plus_hash
from evoeval.eval_test import (
    FAIL,
    PASS,
    CustomEncoder,
    compatible_eval_result,
    estimate_pass_at_k,
    untrusted_check,
)

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]


def get_groundtruth(
    problems, hashcode, use_raw_inputs=False, compute_plus_inputs=False
) -> Dict[str, Any]:
    if hashcode is not None:
        cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
        if os.path.exists(cache_file):
            print(f"Load from ground-truth from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    print("Computing expected output...")
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        with contextlib.redirect_stdout(None):
            oracle["base"], oracle["base_time"] = trusted_exec(
                problem["prompt"] + "\n" + problem["canonical_solution"],
                problem["base_input"]
                if use_raw_inputs
                else [
                    eval(f"[{i}]") for i in problem["inputs"]
                ],  # why do we do this? we have more complex input types.
                problem["entry_point"],
                record_time=True,
                output_not_none=False,
            )
        expected_output[task_id] = oracle

        if compute_plus_inputs:
            oracle["plus"], oracle["plus_time"] = trusted_exec(
                problem["prompt"] + "\n" + problem["canonical_solution"],
                problem["plus_input"],  # assumption: we have plus_input
                problem["entry_point"],
                record_time=True,
                output_not_none=False,
            )
            expected_output[task_id] = oracle

    # print(expected_output)
    print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    if hashcode is not None:
        with open(cache_file, "wb") as f:
            pickle.dump(expected_output, f)

    return expected_output


def check_correctness(
    dataset: str,
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    fast_check=False,
    identifier=None,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
    use_raw_inputs=False,
    compute_plus_inputs=False,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)

    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }

    ret["result"] = untrusted_check(
        dataset,
        solution,
        problem["base_input"]
        if use_raw_inputs
        else [eval(f"[{i}]") for i in problem["inputs"]],
        problem["entry_point"],
        task_id=problem["task_id"],
        expected=expected_output["base"],
        atol=0,  # TODO check
        ref_time=expected_output["base_time"],
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )

    if compute_plus_inputs:
        ret["plus"] = untrusted_check(
            dataset,
            solution,
            problem["plus_input"],
            problem["entry_point"],
            task_id=problem["task_id"],
            expected=expected_output["plus"],
            atol=0,  # TODO check
            ref_time=expected_output["plus_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )

    return ret


def evaluate(flags):
    if flags.parallel is None:
        n_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        n_workers = flags.parallel

    if os.path.isdir(flags.samples):
        result_path = os.path.join(flags.samples, "eval_results.json")
    else:
        assert flags.samples.endswith(".jsonl")
        result_path = flags.samples.replace(".jsonl", "_eval_results.json")

    compute_plus_inputs = False

    if os.path.isfile(result_path) and not flags.i_just_wanna_run:
        print(f"Load from previous results from {result_path}")
        with open(result_path, "r") as f:
            results = json.load(f)

        results = compatible_eval_result(results)
    else:
        use_raw_inputs = False
        if flags.dataset == "humaneval":
            use_raw_inputs = True
            compute_plus_inputs = True
            problems = get_human_eval_plus()
            expected_output = get_groundtruth(
                problems,
                None,
                use_raw_inputs=use_raw_inputs,
                compute_plus_inputs=compute_plus_inputs,
            )
        elif "verbose" in flags.dataset or "concise" in flags.dataset:
            use_raw_inputs = True
            compute_plus_inputs = True
            problems = get_evo_eval(flags.dataset)
            expected_output = get_groundtruth(
                problems,
                None,
                use_raw_inputs=use_raw_inputs,
                compute_plus_inputs=compute_plus_inputs,
            )
        else:
            problems = get_evo_eval(flags.dataset)
            dataset_hash = get_evo_eval_plus_hash(flags.dataset)
            expected_output = get_groundtruth(
                problems,
                dataset_hash,
                use_raw_inputs=use_raw_inputs,
                compute_plus_inputs=compute_plus_inputs,
            )

        results = {
            "eval": {},
        }

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)  # task_id ->
            remainings = set()

            print("Reading samples...")
            for sample in tqdm(load_solutions(flags.samples)):
                task_id = sample["task_id"]
                solution = (
                    sample["solution"]
                    if "solution" in sample
                    else problems[task_id]["prompt"] + sample["completion"]
                )
                remainings.add(sample["_identifier"])
                args = (
                    flags.dataset,
                    completion_id[task_id],
                    problems[task_id],
                    solution,
                    expected_output[task_id],
                    not flags.test_details,  # fast_check
                    sample["_identifier"],
                    flags.min_time_limit,
                    flags.gt_time_limit_factor,
                    use_raw_inputs,
                    compute_plus_inputs,
                )
                futures.append(executor.submit(check_correctness, *args))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remainings), "Missing problems in unfinished"
            assert len(completion_id) == len(problems), "Missing problems in samples"

            def stucking_checker():
                while remainings:
                    last_size = len(remainings)
                    time.sleep(20)
                    if last_size != len(remainings) or len(remainings) == 0:
                        continue
                    # Potentially stuck
                    warn("No samples had finished testing in the last 20s")
                    warn(f"{len(remainings)} samples to be tested: {remainings}")

            threading.Thread(target=stucking_checker).start()

            for future in tqdm(as_completed(futures), total=n_samples):
                result = future.result()
                remainings.remove(result["_identifier"])
                eval_results[result["task_id"]].append(result)

        # sort the results for each problem by completion_id
        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = []
            for res in task_results:

                def get_failed_tests(stat, details, inputs) -> List[Any]:
                    if stat == PASS or not details:
                        return []

                    # if flags.test_details:
                    return [inputs[i] for i in range(len(details)) if not details[i]]

                base_stat, base_details = res["result"]
                base_fail_tests = get_failed_tests(
                    base_stat,
                    base_details,
                    problems[task_id]["base_input"]
                    if use_raw_inputs
                    else [eval(f"[{i}]") for i in problems[task_id]["inputs"]],
                )

                # initialize plus tests
                plus_stat = None
                plus_fail_tests = []

                # with plus tests
                if not flags.base_only and compute_plus_inputs:
                    plus_stat, plus_details = res["plus"]
                    plus_fail_tests = get_failed_tests(
                        plus_stat, plus_details, problems[task_id]["plus_input"]
                    )

                results["eval"][task_id].append(
                    {
                        "task_id": task_id,
                        "solution": res["solution"],
                        "base_status": base_stat,
                        "plus_status": plus_stat,
                        "base_fail_tests": base_fail_tests,
                        "plus_fail_tests": plus_fail_tests,
                    }
                )

    if os.path.isfile(result_path) and flags.i_just_wanna_run:
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(
                results, f, cls=CustomEncoder
            )  # handle some unique cases where failure inputs are sets

        # Calculate pass@k.
    total = np.array([len(r) for r in results["eval"].values()])
    correct = []
    plus_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        correct.append(bc)
        if not flags.base_only and compute_plus_inputs:
            plus_correct.append(
                sum(
                    [
                        res[i]["base_status"] == res[i]["plus_status"] == PASS
                        for i in range(len(res))
                    ]
                )
            )

    correct = np.array(correct)
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in [1, 10, 100]
        if total.min() >= k
    }
    cprint(f"{flags.dataset}", "red")
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.3f}", "red")

    if plus_correct:
        cprint(f"{flags.dataset}+ (base + extra tests)", "green")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(plus_correct), k).mean()
            for k in [1, 10, 100]
            if (total >= k).all()
        }
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.3f}", "green")


def main():
    parser = argparse.ArgumentParser(description="Evaluator")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--samples", required=True, type=str)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--parallel", default=None, type=int)
    parser.add_argument("--i-just-wanna-run", action="store_true")
    parser.add_argument("--test-details", action="store_true")
    parser.add_argument("--min-time-limit", default=1, type=float)
    parser.add_argument("--gt-time-limit-factor", default=4.0, type=float)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument(
        "--noextreme", action="store_true", help="Omit extreme test inputs"
    )
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
