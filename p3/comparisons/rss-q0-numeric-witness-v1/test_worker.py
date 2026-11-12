import unittest,io,tarfile,hashlib,time
import numpy as np
import worker as W

class Tests(unittest.TestCase):
    def pack(self,name='formal/native-evidence/case-0032/query-0000-input.npz'):
        raw=b'abcd';h=tarfile.TarInfo(name);h.size=len(raw);header=h.tobuf(format=tarfile.USTAR_FORMAT)
        ref={'archive':{'uri':'redacted-1e5c8be00b69','bytes':10000},'target':{'member':name,'offset_header':0,'offset_data':512,'bytes':4,'sha256':hashlib.sha256(raw).hexdigest()}}
        class R:
            def get(self,a,s,n):return header if s==0 else raw
        return R(),ref
    def test_exact_member(self):
        r,f=self.pack();self.assertEqual(W.read_member(r,f),b'abcd')
    def test_wrong_member(self):
        r,f=self.pack();f['target']['member']='formal/native-evidence/case-9999/query-0000-input.npz'
        with self.assertRaisesRegex(ValueError,'actual_tar_member'):W.read_member(r,f)
    def test_bridge_rejected(self):
        r,f=self.pack('formal/native-bridge/query-0000-input.npz')
        with self.assertRaisesRegex(ValueError,'actual_case_not_bridge'):W.read_member(r,f)
    def test_bad_payload_SHA(self):
        r,f=self.pack();f['target']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'exact_native_input_SHA'):W.read_member(r,f)
    def test_numeric_and_pixels(self):
        a={'x':np.array([1.,2.]),'img':np.zeros((2,2,3),dtype=np.uint8)};b={'x':np.array([1.,2.25]),'img':a['img'].copy()};b['img'][0,1,2]=3
        d=W.compare(a,b);self.assertEqual(d['x']['changed_elements'],1);self.assertEqual(d['x']['max_abs'],.25);self.assertEqual(d['img']['changed_pixels'],1);self.assertEqual(d['img']['changed_pixel_fraction'],.25)
    def test_ABI_refusal(self):
        with self.assertRaisesRegex(ValueError,'ABI'):W.compare({'x':np.zeros(2)},{'x':np.zeros(3)})
    def test_wrong_array_identity(self):
        a={k:np.zeros(2)for k in W.MAPPING.values()};a.update(sensor_rng_keys=np.zeros(2,dtype=np.uint32),sensor_rng_cached_gaussian=np.array(0.))
        fields={f:W.array_identity(a[k])for f,k in W.MAPPING.items()};fields['sensor_numpy_rng']={'keys':W.array_identity(a['sensor_rng_keys']),'cached_gaussian':W.array_identity(a['sensor_rng_cached_gaussian'])}
        f=io.BytesIO();np.savez(f,**a);self.assertEqual(set(W.load_arrays(f.getvalue(),fields)),set(a));fields['physical_q0']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'native_array_field_identity'):W.load_arrays(f.getvalue(),fields)
    def test_HTTP200_no_read(self):
        r=W.RangeReader({'redacted-1e5c8be00b69':'removed-private-link'},time.time()+60)
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def read(self,*a):raise AssertionError('must_not_read_full_object')
        class Opener:
            def open(self,*a,**k):return Response()
        r.opener=Opener()
        with self.assertRaisesRegex(ValueError,'range_206'):r.get({'uri':'redacted-1e5c8be00b69','bytes':10000},0,512)
        self.assertEqual(r.bytes,0)

if __name__=='__main__':unittest.main()
