"""Bounded metadata, original cursor/loop, tiny native Optax/Orbax checks."""
import copy
import dataclasses
import os
from pathlib import Path
import tempfile
import types
import unittest

import continuation_contract as C
from lr_schedule import ContinuationSchedule
import runtime_adapter as A
import source_loader as S
import temporal_tail as T

def sources():
    loc=C.read(os.environ.get('T30_PARENT_LOCATORS',C.HERE/'PARENT_LOCATORS-v1.json'))
    return loc,S.configure(loc['frozen_repo_root'])

def fixture(seed=0,branch='floor'):
    loc,_=sources()
    row=loc['parents'][seed]
    parent=C.read(row['parent_contract_path'])
    bundle=C.load_bundle(row['metadata_directory'],row['outer_commit'],parent)
    tail=C.build(parent,bundle,f'T30-{branch}-seed{seed}-synthetic-test',branch,
        scientific_attempt_id=f'T30-{branch}-seed{seed}-synthetic-attempt',
        decision_identity={'test_only':True},source_identity=C.identity(C.HERE/'SOURCE_MANIFEST-v1.json'))
    return parent,bundle,tail

class BaseSchedule:
    epochs=50;total_updates=10000;synthetic=True;anchors_per_batch=16;batches_per_epoch=200
    slot_ids=('slot',)
    def __init__(self,parent,bundle):
        self.sha256=parent['temporal_schedule_canonical_sha256']
        self._document={k:bundle['temporal_binding'][k] for k in ('source_support_manifest_sha256','physical_payload_manifest_sha256')}
    def resolve(self,epoch,slot):
        C.need(0<=epoch<50,'physical_epoch_bounds')
        return dict(epoch=epoch,slot=slot,schedule_sha256=self.sha256)
    def epoch_sha(self,epoch):
        return C.digest(self.resolve(epoch,'slot'))

class Tests(unittest.TestCase):
    def test_actual_T20_metadata_and_four_independent_contracts(self):
        identities=[]
        for seed in (0,1):
            for branch in ('floor','reheat'):
                p,b,t=fixture(seed,branch)
                self.assertEqual(len(b['manifest']['files']),34+seed)
                self.assertEqual(b['cursor']['completed_updates'],20000)
                self.assertEqual(t['mandatory_checkpoint_steps'],[25000,30000])
                self.assertEqual(t['AdamW'],p['AdamW'])
                self.assertEqual(t['LR'],p['LR'])
                self.assertEqual(t['total_row_exposures'],960000)
                identities.append(C.digest(t))
        self.assertEqual(len(set(identities)),4)

    def test_incomplete_or_wrong_parent_rejected(self):
        for field in ('step','seed','EMA_only','cursor','next_input','state','branch_ancestry'):
            p,b,t=fixture();b=copy.deepcopy(b)
            if field=='step':b['commit']['effective_updates']=15000
            if field=='seed':b['commit']['training_seed']=1
            if field=='EMA_only':b['manifest']['files']=[r for r in b['manifest']['files'] if not r.get('stock_relative','').startswith('train_state/')]
            if field=='cursor':b['cursor']['completed_updates']=0
            if field=='next_input':b['next_input']={}
            if field=='state':b['state']['tree_sha256']='0'*64
            if field=='branch_ancestry':b['migration']['parent_commit']['sha256']='0'*64
            with self.assertRaises(ValueError):C.validate_parent(p,b)

    def test_cursor_migration_preserves_position_and_parent(self):
        p,b,t=fixture();before=C.canonical(b)
        OLD=sources()[1].old
        old_cursor=OLD.cursor.Cursor(p,b['cursor'])
        cursor=A.migrate_cursor(OLD,p,t,b,old_cursor,b['local'])
        self.assertEqual((cursor.completed,cursor.epoch,cursor.offset),(20000,100,0))
        self.assertEqual(C.canonical(b),before)
        with self.assertRaises(ValueError):OLD.cursor.Cursor(t,b['cursor'])
        self.assertEqual(cursor.order(),OLD.cursor.Cursor(t,cursor.snapshot()).order())

    def test_all_50_physical_epochs_repeat_and_terminal_binding(self):
        p,b,t=fixture();base=BaseSchedule(p,b);schedule=T.TailSchedule(base,t)
        for epoch in range(100,150):
            self.assertEqual(schedule.resolve(epoch,'slot'),base.resolve(epoch%50,'slot'))
        for epoch in (0,50,99,150):
            with self.assertRaises(ValueError):schedule.physical_epoch(epoch)
        cursor=copy.deepcopy(b['cursor']);cursor.update(contract_sha256=C.digest(t),completed_updates=30000,next_epoch=150)
        binding=schedule.checkpoint_binding(cursor,30000,active_reader_epoch=149)
        self.assertIsNone(binding['next_physical_epoch'])
        self.assertEqual(schedule.verify_restore(binding,cursor,30000),(150,0))

    def test_output_cannot_overlap_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);step=root/'parent'/'checkpoint'/'20000';side=root/'parent'/'sidecar'/'20000'
            step.mkdir(parents=True);side.mkdir(parents=True)
            resume=dict(step_directory=str(step),sidecar_directory=str(side))
            parts={'config':types.SimpleNamespace(checkpoint_dir=root/'new-stock')}
            A.validate_output_roots(parts,resume,root/'new-output',root/'new-sidecars')
            for bad in (step,step/'child',step.parent,side,side.parent):
                with self.assertRaises(ValueError):A.validate_output_roots(parts,resume,bad,root/'new-sidecars')

    def test_runtime_only_adds_two_observers(self):
        OLD=sources()[1].old
        ns=dict(OLD.runtime.run.__globals__)
        ns.update(_T30_audit_tx=lambda *_:None,_T30_state_monitor=lambda *_:None)
        adapted=A.adapted_code(OLD.runtime.run,ns)
        self.assertEqual(__import__('inspect').signature(adapted),__import__('inspect').signature(OLD.runtime.run))

    def test_schedule_and_callable_state_tree(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        self.assertEqual(jax.default_backend(),'cpu')
        original=optax.warmup_cosine_decay_schedule(init_value=2.5e-5/1001,peak_value=2.5e-5,
            warmup_steps=1000,decay_steps=10000,end_value=2.5e-6)
        floor=ContinuationSchedule('floor').create();reheat=ContinuationSchedule('reheat').create()
        for step in (0,999,1000,10000,19999,20000,20500,29999,30000):
            self.assertEqual(np.asarray(floor(step)).tobytes(),np.asarray(original(step)).tobytes())
        for step,wanted in ((20000,2.5e-6),(20500,2.5e-5),(30000,2.5e-6)):
            self.assertAlmostEqual(float(reheat(step)),wanted,delta=2e-12)
        self.assertGreaterEqual(float(reheat(29999)),float(reheat(30000)))
        p=jnp.array([1.,-2.],dtype=jnp.float32)
        txs=[optax.chain(optax.clip_by_global_norm(1.),optax.adamw(lr,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10)) for lr in (original,floor,reheat)]
        self.assertEqual(jax.tree.structure(txs[0].init(p)),jax.tree.structure(txs[2].init(p)))
        scalar=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(2.5e-6,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10))
        self.assertNotEqual(jax.tree.structure(txs[0].init(p)),jax.tree.structure(scalar.init(p)))

    def test_Orbax_restore_keeps_moments_counts_EMA_and_new_static_tx(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        import orbax.checkpoint as ocp
        from flax import struct
        @struct.dataclass
        class TinyState:
            step:object
            params:object
            opt_state:object
            ema_params:object
            rng:object
            tx:object=struct.field(pytree_node=False)
        p=jnp.asarray([.3,-.2],jnp.float32);g=jnp.asarray([.4,-.7],jnp.float32)
        def tx(branch):
            return optax.chain(optax.clip_by_global_norm(1.),optax.adamw(ContinuationSchedule(branch).create(),b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10))
        oldtx=tx('floor');opt=oldtx.init(p)
        for _ in range(3):_,opt=oldtx.update(g,opt,p)
        opt=jax.tree.map(lambda x:jnp.asarray(20000,jnp.int32) if x.shape==() and x.dtype==jnp.int32 else x,opt)
        parent=TinyState(jnp.asarray(20000,jnp.int32),p,opt,p*.9,jax.random.key_data(jax.random.key(1)),oldtx)
        before=[np.asarray(x).tobytes() for x in jax.tree.leaves(parent)]
        with tempfile.TemporaryDirectory() as tmp:
            checkpointer=ocp.PyTreeCheckpointer()
            try:
                target=Path(tmp)/'parent20k';checkpointer.save(target,parent)
                for branch in ('floor','reheat'):
                    newtx=tx(branch)
                    template=TinyState(jnp.asarray(0,jnp.int32),p*0,newtx.init(p),p*0,parent.rng*0,newtx)
                    restored=checkpointer.restore(target,item=template)
                    self.assertIs(restored.tx,template.tx)
                    self.assertEqual([np.asarray(x).tobytes() for x in jax.tree.leaves(restored)],before)
                    cursor=types.SimpleNamespace(completed=20000)
                    proof=A.audit_tx(restored,template,cursor,types.SimpleNamespace(lr_schedule=ContinuationSchedule(branch)))
                    self.assertEqual([r['count'] for r in proof['optimizer_counts']],[20000,20000])
                    # The same gradient yields identical next moments regardless of LR.
                    for count in (20000,20500):
                        op=jax.tree.map(lambda x:jnp.asarray(count,jnp.int32) if x.shape==() and x.dtype==jnp.int32 else x,restored.opt_state)
                        updated,next_opt=restored.tx.update(g,op,restored.params)
                        baseline,base_opt=oldtx.update(g,op,restored.params)
                        self.assertEqual([np.asarray(x).tobytes() for x in jax.tree.leaves(next_opt)],
                                         [np.asarray(x).tobytes() for x in jax.tree.leaves(base_opt)])
                        if branch=='floor':self.assertEqual(np.asarray(updated).tobytes(),np.asarray(baseline).tobytes())
                        if branch=='reheat' and count==20500:
                            np.testing.assert_allclose(np.asarray(updated),np.asarray(baseline)*10,rtol=1e-6,atol=1e-12)
            finally:
                checkpointer.close()

    def test_new_tx_uses_preserved_nonzero_moments_and_global_counter(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        self.assertEqual(jax.default_backend(),'cpu')
        p=jnp.asarray([.3,-.2],jnp.float32);g=jnp.asarray([.4,-.7],jnp.float32)
        def tx(branch):
            return optax.chain(optax.clip_by_global_norm(1.),optax.adamw(ContinuationSchedule(branch).create(),b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10))
        old=tx('floor');state=old.init(p)
        for _ in range(3):_,state=old.update(g,state,p)
        state=jax.tree.map(lambda x:jnp.asarray(20500,jnp.int32) if x.shape==() and x.dtype==jnp.int32 else x,state)
        before=[np.asarray(x).tobytes() for x in jax.tree.leaves(state)]
        restored=jax.tree.map(lambda x:np.asarray(x).copy(),state)
        self.assertEqual([np.asarray(x).tobytes() for x in jax.tree.leaves(restored)],before)
        baseline,baseline_state=old.update(g,state,p)
        hot,hot_state=tx('reheat').update(g,restored,p)
        self.assertEqual([np.asarray(x).tobytes() for x in jax.tree.leaves(hot_state)],
                         [np.asarray(x).tobytes() for x in jax.tree.leaves(baseline_state)])
        self.assertEqual([int(x) for x in jax.tree.leaves(hot_state) if x.shape==() and x.dtype==jnp.int32],[20501,20501])
        np.testing.assert_allclose(np.asarray(hot),np.asarray(baseline)*10,rtol=1e-6,atol=1e-12)
        self.assertEqual([np.asarray(x).tobytes() for x in jax.tree.leaves(state)],before)

if __name__=='__main__':
    unittest.main(verbosity=2)
