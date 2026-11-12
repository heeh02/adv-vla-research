import io,json,unittest,tempfile
from pathlib import Path
import read_summaries as R
class Stream(io.BytesIO):
 status=200
 def getheader(self,name):return str(len(self.getvalue()))if name=='Content-Length'else None
class Client:
 def __init__(self,raw):self.raw=raw;self.gets=0
 def get_object_meta_data(self,*a):raise AssertionError('HEAD_forbidden')
 def get_object(self,*a):self.gets+=1;return type('Response',(),{'data':Stream(self.raw)})()
class Tests(unittest.TestCase):
 def test_exact403_authority_and71_existing(self):
  p,_=R.verify();self.assertEqual(len(p['missing403']),403);self.assertEqual(len(p['reuse71']),71)
  self.assertEqual(sum(r['summary']['bytes']for r in p['missing403']),1270348)
 def test_original_GET_body_exact_size_no_HEAD_and_SHA_reject(self):
  p,_=R.verify();fetch=R.original_exact_GET(p);raw=b'{"synthetic":true}';ref=R.identity(raw);client=Client(raw)
  self.assertEqual(fetch(client,R.PREFIX+'cases/p01136/CASE_SUMMARY-v1.json',len(raw),expected=ref),raw)
  self.assertEqual(client.gets,1)
  with self.assertRaises(ValueError):fetch(client,R.PREFIX+'cases/p01136/CASE_SUMMARY-v1.json',len(raw),expected={**ref,'sha256':'0'*64})
 def test_known_local_summary_keeps_original_model_and_OUTCOME_identity(self):
  p,d=R.verify();item=p['reuse71'][0];raw=R.exact(item['local']).read_bytes()
  s=R.validate_summary(raw,item,p,d['final_readback']['model']);self.assertIn('/OUTCOME-v1.json',s['outcome']['uri'])
  with self.assertRaises(ValueError):R.validate_summary(raw,{**item,'success':not item['success']},p,d['final_readback']['model'])
  with tempfile.TemporaryDirectory()as temp:
   out=Path(temp);self.assertIsNone(R.cached(out,item,p,d['final_readback']['model']))
   (out/(item['summary']['sha256']+'-CASE_SUMMARY-v1.json')).write_bytes(raw)
   reused=R.cached(out,item,p,d['final_readback']['model']);self.assertTrue(reused['reused_local'])
   self.assertEqual(reused['outcome'],s['outcome'])
if __name__=='__main__':unittest.main()
