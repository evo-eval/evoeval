# largely adopted from EvalPlus

import ast
import os
import pathlib

from evalplus.data import get_human_eval_plus
from tqdm import tqdm

from evoeval.data import get_evo_eval

INCODER_EXTRA = ["</code>", "<|", "</CODE>"]
POLYCODER_EXTRA = ["\n//", "\n/*"]
NON_CODE_EOFS = ["<|endoftext|>", "\n```", "\n</s>", "\n#"]


def get_all_python_files(folder):
    # return a list of full-path python files
    py_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


def remove_unindented_lines(code, ok_starts):
    new_code = ""
    for line in code.splitlines():
        if any([line.startswith(t) for t in ok_starts]) or line.strip() == "":
            new_code += line + "\n"
            continue

        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            continue

        new_code += line + "\n"

    return new_code


def extract_function(code, target_func):
    def remove_last_line_until_parse(code):
        try:
            tree = ast.parse(code)
        except:
            if "\n" in code:
                code = code.rsplit("\n", 1)[0]
                return remove_last_line_until_parse(code)
            else:
                return None
        return tree

    tree = remove_last_line_until_parse(code)
    if tree is None:  # fail to parse
        return ""

    # return the target function only
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == target_func:
                return ast.unparse(node)
    return ""


def to_four_space_indents(old_code):
    new_code = ""
    for line in old_code.splitlines():
        lspace = len(line) - len(line.lstrip())
        if lspace == 3:
            new_code += " "
        new_code += line + "\n"
    return new_code


def sanitize_folder(args, folder):
    # task_id -> entry_point
    entry_point = {}
    prompts = {}

    if args.dataset == "humaneval":
        problems = get_human_eval_plus()
    else:
        problems = get_evo_eval(args.dataset)

    for task_id, problem in problems.items():
        entry_point[task_id] = problem["entry_point"]
        prompts[task_id] = problem["prompt"]

    # make a new folder with "-sanitized" suffix
    old_folder = pathlib.Path(folder)
    if args.inplace:
        new_folder = old_folder
    else:
        new_folder = old_folder.parent / (old_folder.name + "-sanitized")

    nsan = 0
    ntotal = 0
    for pyf in tqdm(get_all_python_files(folder)):
        # Get [?] from "[prefix]/HumanEval_[?]/[number].py":
        task_id = pyf.split("/")[-2].replace("_", "/")
        ntotal += 1
        old_code = open(pyf).read()

        def_left = "def " + entry_point[task_id] + "("

        imports = prompts[task_id].split(def_left)[0]
        def_right = def_left.join(prompts[task_id].split(def_left)[1:])

        new_code = imports + def_left + old_code.split(def_left)[-1]
        chunks = new_code.split(def_left)  # imports + def_left + {def_right + impl}

        if len(chunks) == 2:
            new_code = def_left + chunks[-1]  # fn + impl

        if "chatgpt" in folder:
            tmp = ""
            for line in new_code.splitlines():
                if line.strip() == "python":
                    continue
                tmp += line + "\n"
            new_code = tmp

        new_code = to_four_space_indents(new_code)

        if args.eof:
            eof_strs = NON_CODE_EOFS
            if "incoder" in folder:
                eof_strs = eof_strs + INCODER_EXTRA
            if "polycoder" in folder:
                eof_strs = eof_strs + POLYCODER_EXTRA
            if "mistral" in folder:
                eof_strs = eof_strs + [r"</s>"]
            for eof in eof_strs:
                new_code = new_code.split(eof)[0]

        # extract the target function and remove lines that are not indented
        new_code = extract_function(new_code, entry_point[task_id])

        if len(chunks) == 2:
            new_code = chunks[0] + new_code

        # write to new folder
        new_pyf = pyf.replace(str(old_folder), str(new_folder))

        if new_code.strip() != old_code.strip():
            print("Sanitized: ", pyf, "->", new_pyf)
            nsan += 1

        pathlib.Path(new_pyf).parent.mkdir(parents=True, exist_ok=True)
        with open(new_pyf, "w") as f:
            f.write(new_code)

    print(f"Sanitized {nsan} out of {ntotal} files.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--eof", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument(
        "--root_folder",
        action="store_true",
        help="Use if we want to sanitize all folders in the root folder.",
    )

    args = parser.parse_args()

    assert not args.folder.endswith("/")

    if not args.root_folder:
        sanitize_folder(args, args.folder)
    else:
        for folder in os.listdir(args.folder):
            if os.path.isdir(f"{args.folder}/{folder}") and "sanitized" not in folder:
                sanitize_folder(args, f"{args.folder}/{folder}")


if __name__ == "__main__":
    main()
