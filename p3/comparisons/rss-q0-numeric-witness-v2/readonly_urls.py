"""Issue four object-scoped, expiring GET URLs in RAM; never return/log credentials."""
import json,sys
from pathlib import Path
from urllib.parse import urlsplit


def install(root,package):
    plan=json.loads((Path(package)/'prepared-v1/INPUT-SPEC-v1.json').read_bytes())['witness_plan']
    original=root.helpers
    def helpers():
        H,items=original();L,M,Backend,proxy,life,stop,G,P=items
        class ExactReadBackend(Backend):
            def sts(self,acl,seconds):
                env=super().sts(acl,seconds)
                if seconds==7200 and len(acl)==2 and acl[0].get('permission')==['READ']:
                    try:
                        sys.path.insert(0,'/private/tmp/paper3-q0-readonly-sdk-v1')
                        sys.path.insert(0,'/Users/researcher/one_vla_latest_main/src')
                        from one_vla.cloud.profile import CloudProfile
                        from one_vla.cloud.jdcloud.object_store import JdcloudObjectStore
                        s3=JdcloudObjectStore(CloudProfile.from_yaml(Path('/Users/researcher/one_vla_latest_main/configs/cloud/jdcloud-cn-north-1.yaml')))._s3()
                        bos=self.old._bos_client(env);urls={}
                        for w in plan['witnesses']:
                            for kind in ('JD','BOS'):
                                uri=w[kind]['archive']['uri'];u=urlsplit(uri)
                                if kind=='JD':v=s3.generate_presigned_url('get_object',Params={'Bucket':u.netloc,'Key':u.path.lstrip('/')},ExpiresIn=7200,HttpMethod='GET')
                                else:v=bos.generate_pre_signed_url(u.netloc,u.path.lstrip('/'),expiration_in_seconds=7200,protocol='https')
                                urls[uri]=v.decode()if isinstance(v,bytes)else v
                        root.C.need(len(urls)==4,'four_exact_object_URLs')
                        env['ADV_VLA_EXACT_READ_URLS']=json.dumps(urls,separators=(',',':'));urls.clear()
                    except Exception:
                        env.clear();raise ValueError('exact_read_URL_issue_failed')from None
                return env
        return H,(L,M,ExactReadBackend,proxy,life,stop,G,P)
    root.helpers=helpers;root.AUTH.add('ADV_VLA_EXACT_READ_URLS')
