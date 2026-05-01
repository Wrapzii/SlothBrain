import json, urllib.request
BASE='http://127.0.0.1:8000'
TESTS=[
 ('find_projects','/task Find my github directory and list projects in it'),
 ('create_file','/task Create a file named smoke_test.txt in the SlothBrain project directory with the text hello from slothbrain'),
 ('read_file','/task Read the file smoke_test.txt in the SlothBrain project directory'),
 ('append_file','/task Append a new line saying second line to smoke_test.txt in the SlothBrain project directory'),
 ('read_file_again','/task Read the file smoke_test.txt in the SlothBrain project directory'),
]
for name,msg in TESTS:
 req=urllib.request.Request(BASE+'/api/chat',data=json.dumps({'message':msg,'mode':'auto','max_steps':10}).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(req,timeout=120) as r:
  data=json.loads(r.read().decode('utf-8'))
 print('###',name)
 print(json.dumps({'agent':data.get('agent'),'response':data.get('response')}, ensure_ascii=False))
