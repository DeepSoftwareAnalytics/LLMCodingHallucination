
from collections import defaultdict
import datetime
import logging
from pathlib import Path
import os
import sys
import json
import re
import argparse
import subprocess
import numpy as np
import multiprocessing

from tqdm import tqdm


CoderEval_ROOT = "../CoderEval"
N = 1


def clean_docstring(docstring):
    docstring = docstring.strip("\"'").strip()
    docstring = docstring.split("\n\n")[0].strip()
    docstring = docstring.split(":param")[0].strip()
    docstring = docstring.split(":return")[0].strip()
    docstring = re.sub(r"\s*\n\s*[A-Z].*$", "", docstring, re.S)
    docstring = re.sub(r"\s+", " ", docstring)
    docstring = docstring.split(". ")[0]
    return docstring


if __name__ == '__main__':
    train_repos = set()
    for jsonl in Path("/home/wangchong/Workspace/Coder-LSP/CodeSearchNet/resources/data/python/final/jsonl/train").rglob("*.jsonl"):
        for line in jsonl.open("r"):
            d = json.loads(line.strip())
            train_repos.add(d["repo"])
        
    with Path(f"{CoderEval_ROOT}/CoderEval4Python.json").open("r") as f:
        samples = json.load(f)["RECORDS"]

    test_examples = []
    for sample in samples:
        # if sample['project'] in train_repos:
        #     continue
        repo = sample['project']
        file_path = f"{sample['file_path']}"
        
        beg_lineno, end_lineno = int(sample['lineno']), int(sample['end_lineno'])

        content_lines = sample['file_content'].split("\n")
        indent = re.match(r"\s*", content_lines[beg_lineno-1]).group(0)
        code_lines = [content_lines[beg_lineno-1][len(indent):]] + content_lines[beg_lineno:end_lineno]
        code = "\n".join(code_lines)
        context_lines = content_lines[:beg_lineno-1] + [indent + '<PLACEHOLDER>'] + content_lines[end_lineno:]
        context = "\n".join(context_lines)

        line = beg_lineno
        column = len(indent)

        docstr = clean_docstring(sample['docstring'])
        func_name = sample['name']
        signature = re.search(r"def\s+(%s\s*\(.*?\)(\s*->.*?)?:)" % re.escape(func_name), code, re.M|re.S)
        if signature is None:
            continue
        signature = signature.group(1)
        idx = code.index(signature) + len(signature)
        prefix = code[:idx]
        body = "\n".join(line for line in code[idx:].split("\n") if line.strip() != "")
        indent = re.match(r"\s*", body.split("\n")[0]).group(0)
        prefix = f"{prefix}\n{indent}'''{docstr}'''\n"
        signature = signature.strip(" :")


        test_examples.append({
            "prompt": prefix,
            "metadata": {
                "task_id": f"{repo.replace('/', '---')}/{sample['_id']}",
                "ground_truth": code,
                "fpath_tuple": file_path.split("/"),
                "context_start_lineno": line - 1,
                "line_no": len(prefix.split("\n")),
                "function_name": func_name,

                "docstring": docstr,
                "signature": signature,
                "lsp-line": line,
                "lsp-column": column,
                "lsp-context": context,
                "lsp-repo": repo,
                "lsp-path": file_path
            }
        })
    with Path(f"datasets/CoderEval-ALL.jsonl").open("w") as f:
        f.write("\n".join([json.dumps(d) for d in test_examples]))
    
    
    