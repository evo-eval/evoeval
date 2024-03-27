# largely adopted from https://github.com/evalplus/evalplus

import itertools
import json
import multiprocessing
import os
import time
from enum import IntEnum, auto
from multiprocessing import Array, Value
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from evalplus.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)

from evoeval.eval_test._creative_special_oracle import (
    _check_maze,
    _check_path,
    _check_product,
)
from evoeval.eval_test._difficult_special_oracle import (
    _check_difficult_poly,
    _check_insensitive_palindrome,
)
from evoeval.eval_test._he_special_oracle import _poly
from evoeval.eval_test._subtle_special_oracle import _check_poly


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, object):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def compatible_eval_result(results: Dict) -> Dict:
    # compatibility
    for task_results in results["eval"].values():
        # update the "files" field to "nfiles"
        if "files" in task_results and "nfiles" not in task_results:
            task_results["nfiles"] = len(task_results.pop("files"))
    return results


# unbiased estimator from https://github.com/openai/human-eval
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    # TODO: search for any close floats? (other data structures)
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


class DataType(IntEnum):
    Float = auto()
    Bool = auto()
    Int = auto()
    Str = auto()
    Null = auto()
    Tuple = auto()
    List = auto()
    Dict = auto()
    Set = auto()
    Type = auto()
    Unknown = auto()


def get_type(x):
    if x is None:
        return DataType.Null
    elif isinstance(x, bool):
        return DataType.Bool
    elif isinstance(x, int):
        return DataType.Int
    elif isinstance(x, str):
        return DataType.Str
    elif is_floats(x):
        return DataType.Float
    elif isinstance(x, tuple):
        return DataType.Tuple
    elif isinstance(x, list):
        return DataType.List
    elif isinstance(x, dict):
        return DataType.Dict
    elif isinstance(x, set):
        return DataType.Set
    elif isinstance(x, type):
        return DataType.Type
    else:
        return DataType.Unknown


def is_equal(x, y) -> tuple[bool, str]:
    x_type, y_type = get_type(x), get_type(y)
    if x_type != y_type:
        return False, "Type mismatch: {} vs {}".format(str(x_type), str(y_type))

    if x_type in [
        DataType.Int,
        DataType.Bool,
        DataType.Null,
        DataType.Str,
        DataType.Set,
        DataType.Type,
    ]:
        if x == y:
            return True, None
        try:
            error_msg = "INT/BOOL/NULL/ Value mismatch: {} vs {}".format(
                repr(x)[:300], repr(y)[:300]
            )
        except:
            error_msg = "Value mismatch: too large for display"
        return False, error_msg
    elif x_type == DataType.Float:
        if np.allclose(x, y, equal_nan=True, atol=1e-6):  # guard against nan
            return True, None
        else:
            return False, "FLOAT Value mismatch: {} vs {}".format(x, y)
    elif x_type in [DataType.List, DataType.Tuple]:
        if len(x) != len(y):
            return False, "Length mismatch: {} vs {}".format(len(x), len(y))
        for i in range(len(x)):
            equal, msg = is_equal(x[i], y[i])
            if not equal:
                return False, msg
        return True, None
    elif x_type == DataType.Dict:
        if len(x) != len(y):
            return False, "Length mismatch: {} vs {}".format(len(x), len(y))
        for k, v in x.items():
            if k not in y:
                return False, "DICT Value mismatch: key {} in {} but not in {}".format(
                    k, x, y
                )
            equal, msg = is_equal(v, y[k])
            if not equal:
                return False, msg
        return True, None
    else:
        # from IPython import embed
        # embed()
        try:
            if x == y:  # e.g., object comparison
                return True, None
            else:
                return False, "ELSE Value mismatch: {} vs {}".format(x, y)
        except:
            return False, "Unsupported type: {} <-- {}".format(x_type, type(x))


def unsafe_execute(
    dataset: str,
    entry_point: str,
    task_id: str,
    code: str,
    inputs,
    expected: List,
    time_limits,
    atol,
    fast_check,
    stat: Value,
    details: Array,
    progress: Value,
):
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        # Disable functionalities that can make destructive changes to the test.
        # allow only 4GB memory usage
        maximum_memory_bytes = 4 * 1024 * 1024 * 1024
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
        exec_globals = {}
        try:
            with swallow_io():
                exec(code, exec_globals)
                fn = exec_globals[entry_point]
                for i, inp in enumerate(inputs):
                    try:
                        with time_limit(time_limits[i]):
                            out = fn(*inp)
                        exp = expected[i]
                        # TODO, for special oracles, think about how to deal with case where
                        # the function has side affect and changes the input ...
                        # this is true especially for some grid checking stuff
                        # ================================================ #
                        # ============== special oracles ================= #
                        # use task_id and dataset to determine the oracle
                        if (
                            dataset == "humaneval"
                            or "verbose" in dataset
                            or "concise" in dataset
                        ) and task_id == "HumanEval/32":
                            assert abs(_poly(*out, inp)) <= 1e-6

                        # =================== Difficult ================== #
                        elif "difficult" in dataset and task_id == "EvoEval/10":
                            _check_insensitive_palindrome(out, *inp, exp)
                        elif "difficult" in dataset and task_id == "EvoEval/32":
                            _check_difficult_poly(*inp, out, exp)

                        # =================== Creative =================== #
                        elif "creative" in dataset and task_id == "EvoEval/26":
                            _check_maze(*inp, out, exp)
                        elif "creative" in dataset and task_id == "EvoEval/30":
                            _check_path(*inp, out, exp)
                        elif "creative" in dataset and task_id == "EvoEval/69":
                            _check_product(*inp, out, exp)

                        # =================== Subtle ===================== #
                        elif "subtle" in dataset and task_id == "EvoEval/32":
                            _check_poly(*inp, out)

                        # =================== Combine ==================== #

                        # =================== Tool Using ================= #

                        # ============== special oracles ================= #
                        # ================================================ #
                        else:
                            exact_match, _ = is_equal(exp, out)
                            assert exact_match
                    except BaseException:
                        details[i] = False
                        progress.value += 1
                        if fast_check:
                            raise
                        continue

                    details[i] = True
                    progress.value += 1
                stat.value = _SUCCESS
        except BaseException:
            stat.value = _FAILED
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(
    dataset: str,
    code: str,
    inputs: List[Any],
    entry_point: str,
    task_id: str,
    expected,
    atol,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
) -> Tuple[str, np.ndarray]:
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = min(os.getenv("EVOEVAL_TIMEOUT_PER_TASK", 60), sum(time_limits)) + 1
    if not fast_check:
        timeout += 1  # extra time for data collection

    # shared memory objects
    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(inputs))])
    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            dataset,
            entry_point,
            task_id,
            code,
            inputs,
            expected,
            time_limits,
            atol,
            fast_check,
            # return values
            stat,
            details,
            progress,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    details = details[: progress.value]

    if not stat:
        stat = TIMEOUT

    if stat == PASS:
        if len(details) != len(inputs) or not all(details):
            stat = FAIL

    return stat, details


def evaluateb_files(
    dataset: str,
    files: List[str],
    inputs: List,
    expected: List,
    entry_point: str,
    atol: float,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
) -> List[Tuple[str, List[bool]]]:
    ret = []
    # sort files by the id in name (i.e., "../n.py")
    files = sorted(files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    for file in files:
        code = open(file, "r").read()
        stat, det = untrusted_check(
            dataset,
            code,
            inputs,
            entry_point,
            expected=expected,
            atol=atol,
            ref_time=ref_time,
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )
        ret.append((stat, det.tolist()))
    return ret
