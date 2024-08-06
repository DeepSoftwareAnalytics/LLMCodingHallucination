import subprocess
import numpy as np
import multiprocessing
from pathlib import Path
import logging


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


ROOT = str(Path(__file__).parent)

dict_std_nonestd={f"{ROOT}/repos/standalone/neo4j-_meta-deprecated.py":f"{ROOT}/repos/neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py",
                  f"{ROOT}/repos/standalone/neo4j-work-query-unit_of_work.py":f"{ROOT}/repos/neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py",
                  f"{ROOT}/repos/standalone/krake-krake-controller-kubernetes-hooks-on.py":f"{ROOT}/repos/rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py"}


if __name__ == "__main__":
    import sys
    import json

    count_tot = 230
    arg1 = sys.argv[1]
    n = int(sys.argv[2])
    fw = open(arg1+"_out.jsonl", 'w')
    f = open(f"{ROOT}/CoderEval4Python.json", 'r', encoding="utf-8")
    content = f.read()
    f.close()

    import os
    import json

    content_json = json.loads(content)
    listtot = []
    collection = {}

    for l in content_json['RECORDS']:
        collection[l["_id"]] = l
    kk = 0
    project_path = f"{ROOT}/repos/"
    dict_id_file={}
    generate_list = []
    for keyy in collection:
        dictTemp = collection[keyy]
        save_data = project_path + "standalone/" + dictTemp["file_path"].replace(".py", "").replace("/",

                                                                                                    "-") + "-" + \
                    dictTemp["name"] + ".py"
        if save_data in dict_std_nonestd.keys():
            save_data = dict_std_nonestd[save_data]
            if os.path.exists(save_data):
                kk+=1
                dict_id_file[dictTemp["_id"]] = save_data
        elif os.path.exists(save_data):
            kk+=1
            dict_id_file[dictTemp["_id"]]=save_data
        else:
            file_path = dictTemp['file_path']
            if project_path + dictTemp["project"].replace("/",
                                                          "---") == f"{ROOT}/repos/neo4j---neo4j-python-driver":
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

    various=0
    list_double=[]
    with open(arg1, 'r') as fr:
        list_tot_question = fr.readlines()
    list_count_tot = []
    dict_level_tot = {}
    tot_k = []
    for i in range(0, n):
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
        f_save_data = open(dict_id_file[str(ques['_id'])], 'r')
        file_content = f_save_data.read()
        f_save_data.close()
        file_content_list = file_content.split("\n")
        import ast
        tka=0
        ast_file = ast.parse(file_content)
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
            for tti in range(0, n):
                temp_tot_k.append(0.0)
        else:
            temp_tot_k = record_out[level]
        dictTemp["generate_results"] = list_generate_code
        fw.write(json.dumps(dictTemp) + "\n")
        fw.flush()
        for k in range(1, n + 1):
            if n - c < k:
                tot_k[k - 1] += 1.0
                temp_tot_k[k - 1] += 1.0
            else:
                tot_k[k - 1] += (1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
                temp_tot_k[k - 1] += (1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
            print(dictTemp["_id"], n, c, tot_k[k - 1])
        record_out[level] = temp_tot_k
    print(len(generate_list), count_tot)
    print(tot_k)
    for tt in tot_k:
        print(tt * 1.0 / count_tot)
        print("% .2f" % (tt * 1.0 / count_tot * 100) + "%")
    print("finish_overall")
    for key_temp in record_out.keys():
        tot_k_1 = record_out[key_temp]
        num_temp = dict_level_tot[key_temp]
        print(key_temp + "\t" + str(num_temp) + "\n")
        for tt in tot_k_1:
            print(" % .2f" % (tt * 1.0 / num_temp * 100) + "%\t")
        print("\n")
    totcc = 0
    print(len(generate_list))
    print(various)
    for l in list_double:
        print(l)
    fw.close()

