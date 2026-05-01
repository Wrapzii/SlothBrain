import json, urllib.request
BASE='http://127.0.0.1:8000'
msgs=[
 '/task what is my user name',
 '/task what is my computers name',
 '/task check my documents',
 '/task find my github directory and list projects',
 '/task create a file named smoke_test.txt in the SlothBrain project directory with the text hello from slothbrain',
 '/task read the file smoke_test.txt in the SlothBrain project directory',
 '/task edit smoke_test.txt with the text second line',
 '/task read the file smoke_test.txt in the SlothBrain project directory',
]
for m in msgs:
 req=urllib.request.Request(BASE+'/api/chat',data=json.dumps({'message':m,'mode':'auto','max_steps':10}).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(req,timeout=120) as r:
  data=json.loads(r.read().decode('utf-8'))
 print('\nMSG:',m)
 print('AGENT:',data.get('agent'))
 print('RESP:',str(data.get('response'))[:500])
