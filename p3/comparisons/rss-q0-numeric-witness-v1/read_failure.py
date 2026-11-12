"""Read only exact registered terminal job and sanitized failure lines."""
import json,re,time
from pathlib import Path
import root_adapter
P=Path(__file__).resolve().parent/'cpu0-package-v1'
R,d,q=root_adapter.prepare(P);H,(L,M,B,proxy,life,stop,G,A)=R.helpers();record=json.loads((P/'run-v1/REGISTERED-v1.json').read_bytes());auth={};backend=None;result={}
try:
 with H.quiet(),proxy():
  auth=L.read_RAM_auth();backend=B(auth,None);raw=backend.describe(record['job_id']);job=raw.get('job',raw.get('data',raw));proof=R.exact_job(raw,record,life);state=life.release_state(raw);lines=[]
  for pod in job.get('pods',[]):
   assert pod['name'].startswith(record['job_id']+'-')
   response=backend.old.convert(backend.old.low.job._send_job_request(b'POST',b'/',body=json.dumps({'jobId':record['job_id'],'podName':pod['name'],'maxLines':120,'chunkSize':1}),params={'action':'DescribeJobLogs','resourcePoolId':H.POOL,'queueID':H.QUEUE}));response=response.get('data',response)
   for line in response.get('logs',[])[:120]:
    line=str(line)
    if len(line)>2000 or any(t in line.lower()for t in ['authorization','https://','secret','accesskey','access_key','token','password','signature']):continue
    lines.append(line)
  result={'job_id':record['job_id'],'definition':proof,'state':state,'observed_epoch':time.time(),'sanitized_lines':lines,'Create_Stop':0}
finally:
 if backend:backend.close()
 auth.clear()
out=P/'run-v1/TERMINAL-FAILURE-LOG-v1.json'
with out.open('x')as f:json.dump(result,f,indent=2)
print(json.dumps(result))
