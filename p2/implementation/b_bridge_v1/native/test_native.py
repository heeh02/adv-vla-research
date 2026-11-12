"""CPU-only synthetic Flax/JAX integration, no checkpoint or real observation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import jax
import jax.numpy as jnp
import ml_dtypes

HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'src/openpi').is_dir())
sys.path.insert(0,str(REPO/'src'))
import build_overlay
import query as Q
import targets as T


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


G=load('_B_bridge_toy_gemma',HERE/'overlay/gemma.py')


def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert jax.default_backend()=='cpu'
        c=G.Config(width=8,depth=18,mlp_dim=16,num_heads=2,num_kv_heads=1,head_dim=4)
        cls.config=c;cls.block=G.Block(configs=(c,c),parent=None)
        cls.x=jnp.reshape(jnp.arange(80,dtype=jnp.float32)/80+.5,(1,10,8)).astype(jnp.bfloat16)
        cls.k=jnp.ones((1,3,1,4),jnp.bfloat16)
        cls.pos=jnp.arange(10,dtype=jnp.int32)[None]+3
        cls.mask=jnp.ones((1,10,13),bool)
        cls.cond=[None,jnp.ones((1,8),jnp.float32)]
        params=cls.block.init(jax.random.key(4),[None,cls.x],(cls.k,cls.k),cls.pos,cls.mask[:,None],cls.cond,True,sru_capture=True)['params']
        # Nonzero known gate makes a forged capture-only edit observable downstream.
        params['pre_attention_norm_1']['Dense_0']['bias']=params['pre_attention_norm_1']['Dense_0']['bias'].at[16:].set(1)
        cls.params=params
        cls.module=G.Module(configs=(c,c),embed_dtype='bfloat16')
        stacked=jax.tree.map(lambda a:jnp.stack([a]*18),params)
        cls.variables={'params':{'layers':stacked,'final_norm_1':{'Dense_0':{'kernel':jnp.zeros((8,24)), 'bias':jnp.zeros(24)}}}}
        cls.cache=tuple(jnp.stack([cls.k]*18)for _ in range(2))

    def test_source_derivation_and_original_math_not_rewritten(self):
        raw=(HERE/'frozen/gemma.py').read_bytes()
        self.assertEqual((HERE/'overlay/gemma.py').read_bytes(),build_overlay.build(raw))
        original=ast.parse(raw);candidate=ast.parse((HERE/'overlay/gemma.py').read_bytes())
        def node(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        self.assertEqual(ast.dump(node(original,'_gated_residual')),ast.dump(node(candidate,'_gated_residual')))
        for name in ('RMSNorm','Attention','FeedForward','PrefixCaptureBlock','ScanNativeCaptureBlock'):
            self.assertEqual(ast.dump(node(original,name)),ast.dump(node(candidate,name)))
        text=(HERE/'overlay/gemma.py').read_text()
        self.assertNotIn('bridge_product_expression',text)
        self.assertIn('layer_index == 8',text)
        independent=node(candidate,'IndependentBridgeCaptureBlock')
        self.assertNotIn('bridge_target=',ast.unparse(independent))

    def test_scanned_dynamic_main_and_independent_targets_are_actually_consumed(self):
        def run(method,active,target):
            return self.module.apply(self.variables,[None,self.x],self.pos,self.mask,self.cond,
                                     kv_cache=self.cache,w_target=target,w_active=active,method=method)
        main=jax.jit(lambda active,target:run('scan_b_bridge_suffix',active,target))
        other=jax.jit(lambda active,target:run('scan_b_bridge_independent_suffix',active,target))
        unused=self.x*2
        inactive=main(jnp.asarray(False),unused)
        target=jnp.zeros_like(unused).at[:,:,0].set(2).at[:,:,1].set(-1)
        changed=main(jnp.asarray(True),target);independent=other(jnp.asarray(True),target)
        self.assertTrue(exact(changed[0],independent[0]))
        for name in G.BRIDGE_FIELDS:
            self.assertTrue(exact(changed[1][name],independent[1][name]),name)
        self.assertTrue(exact(changed[1]['bridge_W_used'][1],target))
        self.assertTrue(exact(changed[1]['bridge_W_before'][1],inactive[1]['bridge_W_before'][1]))
        self.assertFalse(exact(changed[1]['hidden_after_attention_residual'][1],inactive[1]['hidden_after_attention_residual'][1]))
        self.assertFalse(exact(changed[0],inactive[0]))
        np.testing.assert_array_equal(changed[1]['bridge_active'],[False,True,False])
        np.testing.assert_array_equal(inactive[1]['bridge_active'],[False,False,False])
        own=main(jnp.asarray(True),inactive[1]['bridge_W_before'][1])
        self.assertTrue(exact(own[0],inactive[0]))
        for name in G.SCAN_NATIVE_FIELDS:
            self.assertTrue(exact(own[1][name],inactive[1][name]),name)

    def test_independent_setter_does_not_route_to_main_Block_target_kwarg(self):
        import inspect
        text=inspect.getsource(G.IndependentBridgeCaptureBlock)
        self.assertIn('nn.intercept_methods(intercept)',text)
        self.assertIn("isinstance(context.module, Attention)",text)
        self.assertIn('jax.lax.select',text)
        self.assertNotIn('bridge_target=',text)

    def test_native_dtype_and_selector_ABI_refuses_FP32_target(self):
        with self.assertRaises(ValueError):
            self.module.apply(self.variables,[None,self.x],self.pos,self.mask,self.cond,kv_cache=self.cache,
                              w_target=self.x.astype(jnp.float32),w_active=jnp.asarray(True),method='scan_b_bridge_suffix')

    def test_actual_tau_bits_and_weak_type_restored_without_ideal_time(self):
        tau=np.asarray(.3999999165534973,np.float32)
        for weak in (False,True):
            restored=Q.restore_time(tau,weak)
            self.assertTrue(exact(tau,restored))
            self.assertEqual(bool(restored.weak_type),weak)
            self.assertNotEqual(np.asarray(restored).tobytes(),np.asarray(.4,np.float32).tobytes())

    def test_full_integrator_keeps_original_Euler_and_dynamic_numsteps_expressions(self):
        old=ast.parse((HERE/'frozen/candidate_query.py').read_text())
        new=ast.parse((HERE/'query.py').read_text())
        oldfun=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='_integrate')
        newfun=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='_integrate_inactive')
        def assigns(fun,names):
            return {t.id:ast.dump(n.value)for n in ast.walk(fun)if isinstance(n,ast.Assign)for t in n.targets
                    if isinstance(t,ast.Name)and t.id in names}
        self.assertEqual(assigns(oldfun,('dt','next_x','next_time')),assigns(newfun,('dt','next_x','next_time')))
        single=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='_single')
        self.assertNotIn('while_loop',ast.unparse(single));self.assertNotIn('next_x',ast.unparse(single))

    def test_targets_exact_common_geometry_and_new_baseline_gate(self):
        a=np.full((10,1024),256,np.uint16);b=np.zeros_like(a);after=np.full_like(a,3)
        geometry=T.C.analyze_triplet(a,b,after)
        audit=json.dumps({'analysis_complete':True,'synthetic':True,'geometry':geometry},allow_nan=False).encode()
        selected,receipt=T.qualify_and_build_targets(a,b,after,b,after,b,audit,synthetic=True)
        np.testing.assert_array_equal(selected['none'],b[None])
        np.testing.assert_array_equal(selected['natural_d'],after[None])
        self.assertEqual(selected['common_c'].shape,(1,10,1024))
        self.assertFalse(receipt['single_BF16_delta_cast_used'])
        decoded=T.decode_native_target(selected['common_c'])
        self.assertEqual(decoded.dtype,ml_dtypes.bfloat16)
        np.testing.assert_array_equal(decoded.view(np.uint16),selected['common_c'])
        drift=b.copy();drift[0,0]=1
        with self.assertRaisesRegex(ValueError,'baseline_exact'):
            T.qualify_and_build_targets(a,b,after,b,after,drift,audit,synthetic=True)
        with patch.object(T.sys,'platform','darwin'),self.assertRaisesRegex(ValueError,'Linux_only'):
            T.qualify_and_build_targets(a,b,after,b,after,b,audit)

    def test_frozen_input_contract_dependency_closure_resolves_locally(self):
        module=load('_B_bridge_frozen_input_contract_test',HERE/'frozen/input_contract.py')
        hasher=module.hasher()
        self.assertEqual(Path(hasher.__file__).resolve().parent,HERE/'frozen')
        refs=json.loads((HERE/'FROZEN-SOURCE-REFS.json').read_text())
        for name,ref in refs.items():
            raw=(HERE/'frozen'/name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256'])

    def test_actual_jitted_full_inactive_and_legal_single_diagonal_toy(self):
        mask=jnp.asarray([[True,True,False]])
        positions=jnp.asarray([[0,1,1]],jnp.int32)
        cache=(jnp.ones((18,1,3,1,256),jnp.bfloat16),jnp.full((18,1,3,1,256),.25,jnp.bfloat16))
        prepared=Q.J.Prepared(None,cache,cache,mask,mask,positions,positions,mask)
        noise=jnp.full((1,10,32),.125,jnp.float32)
        vmod=cache[1][8].at[:,:2].set(.5)
        target=jnp.zeros((1,10,1024),jnp.bfloat16)
        selectors=tuple(jnp.asarray(x)for x in Q.Q.selectors('pulse'))
        def toy(model,p,x,t,kv,target=None,active=None,independent=False):
            value=jnp.mean(kv[1][8]).astype(jnp.bfloat16)
            v=(.01*x+.02*t+value).astype(jnp.bfloat16)
            out={name:a[0]for name,a in Q.Q._empty(p).items()}
            selected=jnp.asarray((0,8,17),jnp.int32)
            w=jnp.full((3,1,10,1024),value,jnp.bfloat16)
            residual=jnp.broadcast_to(x[:,:,:1],(3,1,10,1024)).astype(jnp.bfloat16)
            out.update(prefix_key_used=kv[0][selected],prefix_value_used=kv[1][selected],
                       attention_output_post_WO_pre_gate=w,receiver_residual_input=residual,
                       hidden_after_attention_residual=residual+w,hidden_post_block=residual+w,
                       attention_gate=jnp.ones((3,1,1,1024),jnp.bfloat16))
            if target is not None:
                out.update(bridge_W_before=w,bridge_W_used=w,bridge_active=jnp.zeros(3,bool))
            return v,out
        with patch.object(Q.J,'_deps',lambda:(jax,jnp,None,None)),patch.object(Q.Q,'_velocity',toy),\
             patch.object(Q,'_velocity',toy),patch.object(Q.Q,'_prepare',lambda *a:(prepared,{})):
            old=jax.jit(lambda n,vm,ns:Q.Q._integrate(None,prepared,n,vm,*selectors,num_steps=ns))
            new=jax.jit(lambda n,vm,wt,wa,ns:Q._integrate_inactive(None,prepared,n,vm,*selectors,wt,wa,num_steps=ns))
            prior=old(noise,vmod,10);candidate=new(noise,vmod,target,jnp.asarray(False),10)
            self.assertTrue(exact(prior[0],candidate[0]))
            for key in ('flow','reads','sampling_abi'):
                for field in prior[1][key]:
                    self.assertTrue(exact(prior[1][key][field],candidate[1][key][field]),(key,field))
            x6=candidate[1]['flow']['x'][6]
            tau=Q.restore_time(candidate[1]['flow']['time'][6],candidate[1]['flow']['time_weak_type'][6])
            single=jax.jit(lambda x,t,w,a,ns:Q.single_query(None,None,None,x,t,w,a,num_steps=ns))
            v,meta=single(x6,tau,target,jnp.asarray(False),10)
            self.assertTrue(exact(v,candidate[1]['flow']['velocity_native'][6]))
            self.assertTrue(exact(meta['actual_x'],x6))
            self.assertTrue(exact(meta['actual_time'],candidate[1]['flow']['time'][6]))
            self.assertTrue(exact(meta['sampling_abi']['actual_dt'],candidate[1]['flow']['dt'][6]))
            self.assertEqual(bool(meta['time_weak_type']),bool(candidate[1]['flow']['time_weak_type'][6]))
            self.assertEqual(int(meta['completed_updates']),0)
            self.assertEqual((int(meta['prefix_prefills']),int(meta['suffix_evaluations'])),(2,1))


if __name__=='__main__':unittest.main()
