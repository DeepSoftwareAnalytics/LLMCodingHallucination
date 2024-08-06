import json
f=open("record_fail_map.json",'r',encoding="utf-8")
content=f.read()
f.close()

dictt_fail_json=json.loads(content)

for keyy in dictt_fail_json.keys():
    print(keyy,len(dictt_fail_json[keyy]))

for l in dictt_fail_json["/home/travis/builds/repos/jaywink---federation"]:
    print(l)

# for l in dictt_fail_json["/home/travis/builds/repos/redhat-openstack---infrared"]:
#     print(l)

# for l in dictt_fail_json["/home/travis/builds/repos/pexip---os-python-cachetools"]:
#     print(l)
#/home/travis/builds/repos/jaywink---federation
# /home/travis/builds/repos/pexip---os-zope 15
# /home/travis/builds/repos/ynikitenko---lena 15
# for l in dictt_fail_json["/home/travis/builds/repos/pexip---os-zope"]:
#     print(l)
