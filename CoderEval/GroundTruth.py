import subprocess
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

tot_success=0
ic=0
zzz=0
dictt_not_project={}
dict_std_nonestd={f"{ROOT}/repos/standalone/neo4j-_meta-deprecated.py":f"{ROOT}/repos/neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py",
                  f"{ROOT}/repos/standalone/neo4j-work-query-unit_of_work.py":f"{ROOT}/repos/neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py",
                  f"{ROOT}/repos/standalone/krake-krake-controller-kubernetes-hooks-on.py":f"{ROOT}/repos/rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py"}
f_map=open(f"{ROOT}/record_fail_map.json",'w',encoding="utf-8")


if __name__ == "__main__":
    c=0
    import sys
    import json

    f = open(f"{ROOT}/testcasesoriginal", 'r', encoding="utf-8")
    content = f.read()
    f.close()
    content_list = content.split("----------------------------")
    dictt = {}
    for i in range(1, len(content_list)):
        temp_content = content_list[i]
        temp_content_list = temp_content.split("\n")
        _id = temp_content_list[1]
        test_content = ""
        for j in range(2, len(temp_content_list)):
            test_content += temp_content_list[j]
            test_content += "\n"
        dictt[_id] = test_content

    # logging.info(dictt)
    # logging.info(len(dictt))
    f = open(f"{ROOT}/CoderEval4Python.json", 'r', encoding="utf-8")
    content = f.read()
    f.close()

    import os
    import json, dill

    content_json = json.loads(content)
    listtot = []
    dictt_origin = {}
    project_dir = "repos"
    for l in content_json['RECORDS']:
        dictt_origin[l["_id"]] = l

    zz = 0
    list_stadnalone_id = []
    for keyy in dictt.keys():
        if dictt[keyy].find(f"{ROOT}/repos/") < 0:
            # zz+=1
            # logging.info("----------------------")
            # logging.info(dictt[keyy])
            if dictt[keyy].find("def ") >= 0:
                list_stadnalone_id.append(keyy)
                zz += 1
    fff = open(f"{ROOT}/record_fail.txt", 'w', encoding="utf-8")
    ffff = open(f"{ROOT}/detailfalut.txt", 'w', encoding="utf-8")
    kk = 0
    project_path = f"{ROOT}/repos/"
    for keyy in dictt_origin:
        dictTemp = dictt_origin[keyy]
        if keyy in list_stadnalone_id:
            save_data = project_path + "standalone/" + dictTemp["file_path"].replace(".py", "").replace("/",

                                                                                                        "-") + "-" + \
                        dictTemp["name"] + ".py"
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
            if save_data.find(f"{ROOT}/repos/mozilla---relman-auto-nag") >= 0:
                save_data = save_data.replace(f"{ROOT}/repos/mozilla---relman-auto-nag/auto-ang",
                                              f"{ROOT}/repos/mozilla---relman-auto-nag/bugbot")

            templ = save_data.split("/")
            if not os.path.exists(save_data):
                logging.info("notexis!!")
                logging.info(save_data)
                continue
        if save_data in dict_std_nonestd.keys():
            save_data=dict_std_nonestd[save_data]
        try:
            process = subprocess.Popen([sys.executable, save_data],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output, error = process.communicate(timeout=30)
        except:
            zzz+=1
            if project_path + dictTemp["project"].replace("/", "---") not in dictt_not_project.keys():
                dictt_not_project[project_path + dictTemp["project"].replace("/", "---")]=[save_data]
            else:
                dictt_not_project[project_path + dictTemp["project"].replace("/", "---")].append(save_data)
            fff.write(save_data+"\n")
            continue

        if process.returncode == 0:
            c += 1
        else:
            ffff.write("filepath:"+save_data+"\n")
            ffff.write("error infoo:\n\n" + bytes.decode(output) + "\n")
            ffff.write("----------------------------------------------")
            ic+=1
            if project_path + dictTemp["project"].replace("/", "---") not in dictt_not_project.keys():
                dictt_not_project[project_path + dictTemp["project"].replace("/", "---")] = [save_data]
            else:
                dictt_not_project[project_path + dictTemp["project"].replace("/", "---")].append(save_data)
            fff.write(save_data + "\n")
    f_map.write(json.dumps(dictt_not_project))
    f_map.close()
    logging.info(c,ic,zzz)
    if os.path.exists(f"{ROOT}/pythonsol_unittest.log"):
        if os.path.isdir(f"{ROOT}/pythonsol_unittest.log"):
            os.removedirs(f"{ROOT}/pythonsol_unittest.log")
        if os.path.isfile(f"{ROOT}/pythonsol_unittest.log"):
            os.remove(f"{ROOT}/pythonsol_unittest.log")
