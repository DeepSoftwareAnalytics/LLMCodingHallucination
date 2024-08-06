
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
import traceback
import numpy as np
import multiprocessing

from tqdm import tqdm

sys.path.append("..")


from make_window import MakeWindowWrapper
from build_vector import BuildVectorWrapper, BagOfWords
from search_code import CodeSearchWrapper
from build_prompt import BuildPromptWrapper
from utils import CONSTANTS, CodexTokenizer

from log_utils import init_log
from model import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Process(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as exception:
            self._cconn.send(exception)

    def join(self, timeout):
        super().join(timeout)

        if self.is_alive():
            self.terminate()
        super().join()

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def clean_docstring(docstring):
    # docstring = docstring.strip("\"'").strip()
    docstring = docstring.split("\n\n")[0].strip()
    docstring = docstring.split(":param")[0].strip()
    docstring = docstring.split(":return")[0].strip()
    docstring = re.sub(r"\s*\n\s*[A-Z].*$", "", docstring, re.S)
    docstring = re.sub(r"\s+", " ", docstring)
    docstring = docstring.split(". ")[0]
    return docstring

def clean_code(code:str):
    lines = code.split("\n")
    lines = [line for line in lines if not line.strip().startswith("#")]
    code = "\n".join(lines)
    return re.sub(r"'''(.*)'''", "", code)

def make_repo_window(repos, window_sizes, slice_sizes):
    worker = MakeWindowWrapper(None, repos, window_sizes, slice_sizes)
    worker.window_for_repo_files()


def run_RG1_and_oracle_method(benchmark, repos, window_sizes, slice_sizes):
    # build code snippets for all the repositories
    make_repo_window(repos, window_sizes, slice_sizes)
    # build code snippets for vanilla retrieval-augmented approach and ground truth
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_baseline_and_ground()
    # build vector for vanilla retrieval-augmented approach and ground truth
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_repo_windows()
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_baseline_and_ground_windows()
    # search code for vanilla retrieval-augmented approach and ground truth
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_baseline_and_ground()
    # build prompt for vanilla retrieval-augmented approach and ground truth
    tokenizer = CodexTokenizer
    mode = CONSTANTS.rg
    output_file_path = f"prompts/rg-one-gram-ws-20-ss-2.jsonl"
    BuildPromptWrapper('one-gram', benchmark, repos, window_sizes[0], slice_sizes[0], tokenizer).build_first_search_prompt(mode, output_file_path)



def run_RepoCoder_method(benchmark, repos, window_sizes, slice_sizes, prediction_path):
    mode = CONSTANTS.rgrg
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_prediction(mode, prediction_path)
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_prediction_windows(mode, prediction_path)
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_prediction(mode, prediction_path)
    tokenizer = CodexTokenizer
    output_file_path = f"prompts/{MODEL_NAME}/repocoder-one-gram-ws-20-ss-2.jsonl"
    BuildPromptWrapper('one-gram', benchmark, repos, window_sizes[0], slice_sizes[0], tokenizer).build_prediction_prompt(mode, prediction_path, output_file_path)


MODEL_FACTORY = {
    "codegen": ("Salesforce/codegen-350M-mono", init_codegen, 2048)
}


CoderEval_ROOT = "CoderEval"
N = 1


def run_test_cases(prediction_path):
   
    with Path(f"{CoderEval_ROOT}/CoderEval4Python.json").open("r") as f:
        samples = json.load(f)["RECORDS"]

    count_tot = len(tasks)
    dict_std_nonestd={f"{CoderEval_ROOT}/repos/standalone/neo4j-_meta-deprecated.py":f"{CoderEval_ROOT}/repos/neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py",
                f"{CoderEval_ROOT}/repos/standalone/neo4j-work-query-unit_of_work.py":f"{CoderEval_ROOT}/repos/neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py",
                f"{CoderEval_ROOT}/repos/standalone/krake-krake-controller-kubernetes-hooks-on.py":f"{CoderEval_ROOT}/repos/rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py"}

    
    fw = open(prediction_path + "_out.jsonl", 'w')
    
    listtot = []
    collection = {sample["_id"]: sample for sample in samples}

    kk = 0
    project_path = f"{CoderEval_ROOT}/repos/"
    dict_id_file={}
    generate_list = []
    for keyy in collection:
        dictTemp = collection[keyy]
        save_data = project_path + "standalone/" + dictTemp["file_path"].replace(".py", "").replace("/", "-") + "-" + \
                    dictTemp["name"] + ".py"
        # logging.info(f"Processing _id: {dictTemp['_id']}")
        # logging.info(f"Initial save_data: {save_data}")
        if save_data in dict_std_nonestd.keys():
            save_data = dict_std_nonestd[save_data]
            if Path(save_data).exists():
                kk += 1
                dict_id_file[dictTemp["_id"]] = save_data
        elif Path(save_data).exists():
            kk+=1
            dict_id_file[dictTemp["_id"]] = save_data
        else:
            file_path = dictTemp['file_path']
            if project_path + dictTemp["project"].replace("/", "---") == f"{CoderEval_ROOT}/repos/neo4j---neo4j-python-driver":
                save_data = os.path.join(project_path + dictTemp['project'].replace("/", "---") + "/src",
                                        file_path).replace(
                    ".py", "_" + dictTemp["name"] + "_passk_validte.py")
            else:
                save_data = os.path.join(project_path + dictTemp['project'].replace("/", "---"), file_path).replace(
                    ".py", "_" + dictTemp["name"] + "_passk_validte.py")
            if save_data in dict_std_nonestd.keys():
                save_data = dict_std_nonestd[save_data]

            if os.path.exists(save_data):
                kk+=1
                dict_id_file[dictTemp["_id"]] = save_data


    with open(prediction_path, 'r') as fr:
        list_tot_question = fr.readlines()
    list_count_tot = []
    dict_level_tot = {}
    tot_k = []
    for i in range(0, N):
        tot_k.append(0.0)
    record_out = {}
    for i in range(0, len(list_tot_question)):
        dictTemp = {}
        ques = json.loads(list_tot_question[i])
        content_doc = collection[ques["_id"]]
        if content_doc is None:
            continue
        dictTemp["file_path"] = content_doc["file_path"]
        if "project" in content_doc.keys():
            dictTemp["project"] = content_doc["project"]
        dictTemp["name"] = content_doc["name"]
        dictTemp["docstring"] = content_doc["docstring"]
        dictTemp["_id"] = str(ques['_id'])
        solutions = ques["generate_results"]
        list_code = []
        for solution in solutions:
            list_code.append(solution)
        dictTemp['code'] = list_code
        level = content_doc["level"]
        dictTemp["level"] = level
        if level not in dict_level_tot.keys():
            dict_level_tot[level] = 1
        else:
            dict_level_tot[level] += 1
        generate_list.append(dictTemp)
        # logging.info(f"dict_id_file: {dict_id_file}")
        # logging.info(f"dict_id_file: {dict_id_file[str(ques['_id'])]}")
        f_save_data = open(dict_id_file[str(ques['_id'])], 'r')
        file_content = f_save_data.read()
        f_save_data.close()
        file_content_list = file_content.split("\n")
        import ast
        tka=0
        ast_file = ast.parse(file_content)
        # logging.info(f"ast_file: {ast_file}")
        start_indent = 0
        new_data = ""
        for node in ast.walk(ast_file):
            if isinstance(node, ast.FunctionDef):
                temp_method_name = node.name
                if content_doc["name"] != temp_method_name and "_"+content_doc["name"]!=temp_method_name:
                    continue
                start_line = node.lineno
                end_line = node.end_lineno
                indent_s = file_content_list[start_line - 1]
                tttt = indent_s.lstrip(" ")
                start_indent = len(indent_s) - len(tttt)
                new_data=""
                for i in range(0, start_line - 1):
                    new_data += file_content_list[i]
                    new_data += "\n"
                new_data += "<insert generated code here>\n"
                for i in range(end_line, len(file_content_list)):
                    new_data += file_content_list[i]
                    new_data += "\n"
        # logging.info(f'new_data: {new_data}')          
        assert new_data!=""
        list_generate_code = []
        c = 0
        code_num = 0
        for code in list_code:
            dict_temp = {}
            dict_temp["generate_code"] = code
            code_list = code.split("\n")
            tttt = code_list[0].lstrip(" ")
            code_indent = len(code_list[0]) - len(tttt)
            new_code = ""
            if start_indent > code_indent:
                str_a = ""
                for iii in range(0, start_indent - code_indent):
                    str_a += " "
                for ccc in code_list:
                    ttz = str_a + ccc
                    new_code += ttz
                    new_code += "\n"
            else:
                new_code = code
            out_data = new_data.replace("<insert generated code here>", new_code)
            save_data_new=dict_id_file[str(ques['_id'])]
            f = open(save_data_new.replace(".py", str(code_num) + ".py"), 'w')
            f.write(out_data)
            f.close()
            try:
                process = subprocess.Popen([sys.executable, save_data_new.replace(".py", str(code_num) + ".py")],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output, error = process.communicate(timeout=30)
            except:
                code_num += 1
                continue
            if process.returncode == 0:
                dict_temp["is_pass"] = True
                c += 1
            else:
                dict_temp["is_pass"] = False
            dict_temp["return_code"] = process.returncode
            code_num += 1
            list_generate_code.append(dict_temp)

        if level not in record_out.keys():
            temp_tot_k = []
            for tti in range(0, N):
                temp_tot_k.append(0.0)
        else:
            temp_tot_k = record_out[level]
        dictTemp["generate_results"] = list_generate_code
        fw.write(json.dumps(dictTemp) + "\n")
        fw.flush()
        for k in range(1, N + 1):
            if N - c < k:
                tot_k[k - 1] += 1.0
                temp_tot_k[k - 1] += 1.0
            else:
                tot_k[k - 1] += (1.0 - np.prod(1.0 - k / np.arange(N - c + 1, N + 1)))
                temp_tot_k[k - 1] += (1.0 - np.prod(1.0 - k / np.arange(N - c + 1, N + 1)))
            logging.info(f'{dictTemp["_id"]} {N} {c} {tot_k[k - 1]}')
        record_out[level] = temp_tot_k
    fw.close()

    logging.info("\n")
    logging.info("\n")
    logging.info(f'## total: {count_tot}')
    for k, tt in enumerate(tot_k, 1):
        logging.info(f"pass@{k}: {round(tt / count_tot * 100, 1)}% ({int(tt)})")
    logging.info("\n")

    for key in ["self_contained", "slib_runnable", "plib_runnable", "class_runnable", "file_runnable", "project_runnable"]:
        tot_k = record_out[key]
        logging.info(f'## {key}: {dict_level_tot[key]}')
        for k, tt in enumerate(tot_k, 1):
            logging.info(f"pass@{k}: {round(tt / dict_level_tot[key] * 100, 1)}% ({int(tt)})")
        logging.info("\n")


def predict(tasks):
    predictions = []
    batch_ranges = list(zip(range(0, len(tasks), BATCH_SIZE), range(BATCH_SIZE, len(tasks)+BATCH_SIZE, BATCH_SIZE)))
    for beg, end in tqdm(batch_ranges, ascii=True, desc="Evaluation"):
        _ids = []
        batch = []
        for task in tasks[beg:end]:
            _id = task["metadata"]["task_id"].split("/")[-1]
            _ids.append(_id)
            batch.append({
                "docstr": clean_docstring(task["metadata"]["docstring"]),
                "signature": task["metadata"]["signature"],
                "prefix": task["prompt"],
            })
        try:
            # outputs = generator.generate_simple(batch, max_len=MAX_LEN, repetition_penalty=REPETITION_PENALTY)
            outputs = generator.generate(batch, max_len=MAX_LEN, repetition_penalty=REPETITION_PENALTY)
        except Exception:
            traceback.print_exc()
            outputs = [f"def {inst['signature']}:\npass" for inst in batch]
        predictions.extend([(_id, clean_code(output)) for _id, output in zip(_ids, outputs)])
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", required=True, type=str)
    parser.add_argument("-max_len", "--max_len", required=False, type=int, default=192)
    parser.add_argument("-batch", "--batch", required=False, type=int, default=4)
    args = parser.parse_args()


    repos = [p.parts[-1] for p in Path("repos").glob("*---*") if p.is_dir()]
    window_sizes = [20]
    slice_sizes = [2]  # 20 / 2 = 10

    
    MODEL = args.model
    MAX_LEN = args.max_len
    REPETITION_PENALTY = 1
    BATCH_SIZE = args.batch
    DEVICE = "cuda"


    # MODEL_NAME, INIT_FUNC, MODEL_MAX_LEN = MODEL_FACTORY[MODEL]

    if MODEL.startswith("gpt-"):
        MODEL_NAME = MODEL
        generator = ChatGPT(MODEL)
    else:
        MODEL_NAME, INIT_FUNC, MODEL_MAX_LEN = MODEL_FACTORY[MODEL]
        model, tokenizer = INIT_FUNC(
            model_name=MODEL_NAME,
            checkpoint=None,
            additional_tokens=[],
            device=DEVICE
        )
        generator = Generator(model, tokenizer, MODEL_MAX_LEN)


    init_log(f"testing-CoderEval/{MODEL}/testing.log", logging.INFO)
    # RAW method experiment result
    PRED_RESULT_FILE_R0 = f"testing-CoderEval/{MODEL}/predictions_r0.jsonl"
    # RAG method experiment result
    PRED_RESULT_FILE_R1 = f"testing-CoderEval/{MODEL}/predictions_r1.jsonl"
    

    tasks = []
    id2task = dict()
    with Path(f"datasets/CoderEval-ALL.jsonl").open("r") as f:
        for line in f:
            task = json.loads(line.strip())
            _id = task["metadata"]["task_id"].split("/")[-1]
            id2task[_id] = task
            tasks.append(task)
    
    
    logging.info(f"model: {MODEL_NAME}")
    logging.info(f"dataset: CoderEval")
    logging.info(f"dataset size: {len(id2task)}")
    logging.info(f"max len: {MAX_LEN}")
    logging.info(f"repeatition penalty: {REPETITION_PENALTY}")

    ## RAW method
    predictions = predict(tasks)
    lines = []
    for _id, pred in predictions:
        task = id2task[_id].copy()
        task["_id"] = _id
        task["choices"] = [{"text": pred}]
        task["generate_results"] = [pred]
        lines.append(json.dumps(task))
    with Path(PRED_RESULT_FILE_R0).open("w") as f:
        f.write("\n".join(lines))
    
    run_test_cases(PRED_RESULT_FILE_R0)

    ## RAG-based method
    # run_RG1_and_oracle_method(CONSTANTS.codereval_benchmark, repos, window_sizes, slice_sizes)
    tasks = []
    with Path(f"prompts/rg-one-gram-ws-20-ss-2.jsonl").open("r") as f:
        for line in f:
            d = json.loads(line.strip())
            _id = d["metadata"]["task_id"].split("/")[-1]
            task = id2task[_id].copy()
            task["prompt"] = d["prompt"]
            tasks.append(task)
    logging.info(len(tasks))
    predictions = predict(tasks)
    logging.info(len(predictions))
    # logging.info(predictions)
    
    lines = []
    for _id, pred in predictions:
        task = id2task[_id].copy()
        task["_id"] = _id
        task["choices"] = [{"text": pred}]
        task["generate_results"] = [pred]
        lines.append(json.dumps(task))
    with Path(PRED_RESULT_FILE_R1).open("w") as f:
        f.write("\n".join(lines))
    
    run_test_cases(PRED_RESULT_FILE_R1)





