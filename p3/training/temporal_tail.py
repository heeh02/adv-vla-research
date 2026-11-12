"""Global 100..149 cursors over unchanged physical 0..49 source bindings."""
import copy
import continuation_contract as C

class TailSchedule:
    def __init__(self,base,tail):
        C.need(tail['arm']=='T' and base.epochs==50 and base.total_updates==10000, 'original_T50_schedule')
        C.need(base.sha256 == tail['continuation']['T_original_schedule_sha256'], 'parent_schedule_identity')
        self.base=base;self.tail_contract_sha256=C.digest(tail)
        self.sha256=base.sha256;self.synthetic=base.synthetic;self._document=base._document
        self.epochs=150;self.total_updates=30000
        self.anchors_per_batch=base.anchors_per_batch;self.batches_per_epoch=base.batches_per_epoch
    @property
    def slot_ids(self): return self.base.slot_ids
    def physical_epoch(self,epoch):
        C.need(type(epoch) is int and 100<=epoch<150,'tail_global_epoch_bounds')
        return epoch%50
    def resolve(self,epoch,slot):
        # EXACT original dict required by qualified EpochBackend.__call__.
        return self.base.resolve(self.physical_epoch(epoch),slot)
    def epoch_sha(self,epoch):
        if epoch==150:return None
        return C.digest(dict(global_epoch=epoch,physical_epoch=self.physical_epoch(epoch),
            original_physical_map_sha256=self.base.epoch_sha(epoch%50),tail_contract_sha256=self.tail_contract_sha256))
    def checkpoint_binding(self,cursor,native_step,*,active_reader_epoch=None):
        c=copy.deepcopy(cursor);step=c['completed_updates']
        C.need(type(step) is int and step==native_step and 20000<=step<=30000,'tail_native_cursor_step')
        C.need(c['contract_sha256']==self.tail_contract_sha256,'tail_cursor_contract')
        epoch,offset=divmod(step,self.batches_per_epoch)
        C.need((c['next_epoch'],c['next_batch_in_epoch'])==(epoch,offset),'global_next_unconsumed_position')
        plan=c['active_epoch_plan']
        C.need((not offset or plan is not None) and (plan is None or plan['epoch']==epoch),'persist_global_epoch_bank')
        C.need(c['active_epoch_plan_sha256']==(C.digest(plan) if plan else None),'global_plan_SHA')
        if active_reader_epoch is not None:
            C.need(type(active_reader_epoch) is int and 100<=active_reader_epoch<150 and (active_reader_epoch==epoch or (offset==0 and active_reader_epoch==epoch-1)),'global_active_reader_position')
        return dict(schema='T30-temporal-next-cursor-binding-v1',tail_contract_sha256=self.tail_contract_sha256,
            original_schedule_sha256=self.base.sha256,native_cursor_sha256=C.digest(c),effective_updates=step,
            next_global_epoch=epoch,next_physical_epoch=None if epoch==150 else self.physical_epoch(epoch),
            next_batch_in_epoch=offset,next_epoch_map_sha256=self.epoch_sha(epoch),
            active_global_reader_epoch=active_reader_epoch,
            active_physical_reader_epoch=None if active_reader_epoch is None else self.physical_epoch(active_reader_epoch),
            source_support_manifest_sha256=self._document['source_support_manifest_sha256'],
            physical_payload_manifest_sha256=self._document['physical_payload_manifest_sha256'],
            consume_only_after_blocked_finite_update=True,redraw_on_restore=False)
    def verify_restore(self,binding,cursor,native_step):
        C.need(binding==self.checkpoint_binding(cursor,native_step,active_reader_epoch=binding['active_global_reader_epoch']),'tail_binding_changed')
        return binding['next_global_epoch'],binding['next_batch_in_epoch']

def production(epoch_module,parts,production,catalog,document,schedule,cache_parent,transport,members,on_cache,verify_batch=None):
    """Same restore/decoder/cache/BatchPipeline; only acquire argument is mapped."""
    legacy=production.clean.LegacyCleanBackend(parts['store'],parts['store'].index,synthetic=False)
    def acquire(global_epoch):
        physical=schedule.physical_epoch(global_epoch)
        index,receipt,owner=production.restore.restore_epoch(production.bridge,catalog,document,physical,
            cache_parent,transport,members,synthetic=False)
        C.need(index['epoch']==physical,'restored_physical_epoch')
        on_cache('restore',global_epoch,dict(global_epoch=global_epoch,physical_epoch=physical,index=index,receipt=receipt,owner=owner))
        backend=production.bridge.EpochBackend(catalog,document,index,C.digest(index),legacy,synthetic=False)
        return backend,(global_epoch,index)
    def release(token,*,prefetch_joined):
        global_epoch,index=token
        result=production.restore.cleanup_epoch(production.bridge,index,prefetch_joined=prefetch_joined)
        on_cache('cleanup',global_epoch,dict(global_epoch=global_epoch,physical_epoch=index['epoch'],result=result))
    return epoch_module.EpochPipeline(schedule,production.temporal,acquire,release,
        lambda view:parts['native_BatchPipeline'](view,parts['data_config'],parts['data_sharding']),verify_batch)
