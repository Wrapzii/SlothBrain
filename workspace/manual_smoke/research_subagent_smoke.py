import json, urllib.request
BASE='http://127.0.0.1:8000'
msgs=[
 '/task what is my computers name',
 '/task research sun hydraulics llc and provide sources',
 '/task use a sub agent to summarize what this repository is for',
]
for m in msgs:
 req=urllib.request.Request(BASE+'/api/chat',data=json.dumps({'message':m,'mode':'auto','max_steps':10}).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(req,timeout=180) as r:
  data=json.loads(r.read().decode('utf-8'))
 print('\nMSG:',m)
 print('AGENT:',data.get('agent'))
 print('RESP:',str(data.get('response'))[:700])
