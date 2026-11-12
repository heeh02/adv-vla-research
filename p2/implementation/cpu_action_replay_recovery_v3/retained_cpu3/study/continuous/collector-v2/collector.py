"""Bounded observations in the original five-slot parent loop; no sensor rewrite."""
from contextlib import contextmanager
import copy
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'native-pairing'),str(HERE.parent)]
import bridge_v2 as B
from bridge import NativePairBridge
from camera_sensor_pair import CAMERAS,NativeCameraPairTap
S=B.S

class ScheduledRecorder:
    def __init__(self,plan,emit):
        self.plan=copy.deepcopy(plan);self.emit=emit;self.root_id=plan['root_trajectory_id'];self.startup=10
        indices=plan['recorded_query_indices'];maximum=plan['max_policy_steps']
        S.need(self.root_id and callable(emit)and type(maximum)is int and maximum>0,'explicit_root_plan')
        S.need(indices and all(type(q)is int and 0<=q<(maximum+4)//5 for q in indices)
               and indices==sorted(set(indices)),'frozen_unique_query_indices')
        S.need(set(plan['tiers'])=={str(q)for q in indices},'one_explicit_tier_per_planned_query')
        self.rows=[{'window_slot':i,'planned_query_index':q,'planned_env_step_index':5*q,'tier':copy.deepcopy(plan['tiers'][str(q)]),'status':'planned'}for i,q in enumerate(indices)]
        self.next=0;self.closed=False;self.planned_receipt=None
        self.planned_receipt=emit('WINDOW_PLANNED',self.summary(),{})
        S.need(isinstance(self.planned_receipt,str)and self.planned_receipt,'durable_plan_before_rollout')
    def summary(self):
        return {'schema':'sru-scheduled-query-window-v2','root_trajectory_id':self.root_id,
                'planned_queries':len(self.rows),'planned_receipt':self.planned_receipt,'rows':copy.deepcopy(self.rows),
                'full_replay_qualified':False,'queries_are_actual_parent_boundaries':True}
    def record(self,*,window_slot,query_index,env_step_index,absolute_env_step_index,query_time_actual,state,pair_supplier):
        S.need(not self.closed and self.next<len(self.rows)and window_slot==self.next,'next_actual_scheduled_slot')
        row=self.rows[self.next]
        S.need((query_index,env_step_index,absolute_env_step_index)==(row['planned_query_index'],row['planned_env_step_index'],10+row['planned_env_step_index']),'actual_producer_query_clock')
        byname={b.name:b for b in state.bindings};clock=byname['root.clock'].snapshot()
        S.need(all(clock[k]==v for k,v in [('query_index',query_index),('env_step_index',env_step_index),('absolute_env_step_index',absolute_env_step_index)]),'bound_actual_clock')
        S.need(S.R._identity(clock['query_time_actual'])==S.R._identity(query_time_actual),'bound_actual_query_time')
        S.need(byname['root.action_buffer'].snapshot()['items']==(),'original_five_slots_all_consumed')
        row.update(status='reached',query_index=query_index,env_step_index=env_step_index,absolute_env_step_index=absolute_env_step_index)
        self.emit('QUERY_REACHED',copy.deepcopy(row),{'query_time_actual':query_time_actual})
        pair,proof=S.same_state_pair(state,pair_supplier,query_env_step=env_step_index,query_time=query_time_actual)
        receipt=self.emit('PAIRED_INPUT',{**copy.deepcopy(row),**proof},{'views':pair,'query_time_actual':query_time_actual})
        S.need(isinstance(receipt,str)and receipt,'durable_actual_pair')
        row.update(status='recorded',pair_receipt=receipt);self.next+=1
        return pair,proof
    def finish(self,reason):
        S.need(not self.closed and reason in ('window_complete','root_terminal','root_failure','deadline'),'one_window_finish')
        S.need(reason!='window_complete'or self.next==len(self.rows),'cannot_complete_missing_slots')
        self.closed=True
        for row in self.rows[self.next:]:row.update(status='unreached',reason=reason)
        value=self.summary();self.emit('WINDOW_CLOSED',value,{});return value

class ScheduledBridge(NativePairBridge):
    """Generalizes only the old frozen 12-window scheduling. Sensor tap reused."""
    def __init__(self,recorder,state,tap,cursor,sensor_gate):
        S.need(isinstance(recorder,ScheduledRecorder)and not recorder.closed and recorder.next==0,'fresh_scheduled_recorder')
        S.need(isinstance(tap,NativeCameraPairTap)and isinstance(cursor,B.IssuedSliceCursor),'original_sensor_and_actual_for_slot_witness')
        self.rec,self.state,self.tap,self.issued=recorder,state,tap,cursor
        self.bindings={b.name:b for b in state.bindings}
        S.need(self.bindings['root.action_buffer']is cursor.binding,'not_a_fabricated_empty_deque')
        S.need(callable(sensor_gate),'actual_observable_provenance_gate_required')
        self.sensor_gate=sensor_gate;self.pending=None;self.failure_in_memory=None;self._in_step=False
    @contextmanager
    def producing_step(self,*,next_n,next_absolute,startup=False,slot=None,action=None):
        S.need(not self.rec.closed and not self._in_step,'one_open_actual_parent_step')
        row=self.rec.rows[self.rec.next]if self.rec.next<len(self.rec.rows)else None
        armed=row is not None and next_n==row['planned_env_step_index']and(not startup or next_absolute==10)
        attempted=self.rec.next if armed else None
        clock=self._clock()
        S.need(next_absolute==clock['absolute_env_step_index']+1,'actual_absolute_step_increment')
        S.need((startup and next_n==0 and 1<=next_absolute<=10)or(not startup and next_n==clock['env_step_index']+1 and next_absolute==next_n+10),'original_dummy_or_policy_step_clock')
        S.need(self.pending is None,'paired_sensor_not_yet_consumed')
        if row is not None:S.need(next_n<=row['planned_env_step_index'],'planned_query_was_skipped')
        @contextmanager
        def actual_scope():
            if startup:yield
            else:
                with self.issued.issuing(slot,action):yield
        try:
            self._in_step=True
            with actual_scope():
                if armed:
                    old={c:self.tap.records.get(c)for c in CAMERAS};before=copy.deepcopy(self.tap.clock_state()['sim_time'])
                    with self.tap.during_parent_step(next_n):yield True
                    after=self.tap.clock_state()['sim_time']
                    S.need(float(after)>=float(before),'actual_time_not_monotonic')
                    for camera in CAMERAS:
                        record=self.tap.records.get(camera)
                        S.need(record is not None and record is not old[camera]and record['sensor_env_step_index']==next_n,'actual_camera_callback_required')
                        S.need(float(before)<=float(record['sensor_time'])<=float(after),'camera_own_time_outside_producing_cycle')
                    self.pending={'window_slot':self.rec.next,'env_step_index':next_n,'records':{c:self.tap.records[c]for c in CAMERAS}}
                else:yield False
                now=self._clock();S.need(now['env_step_index']==next_n and now['absolute_env_step_index']==next_absolute,'producer_must_publish_completed_counts')
        except BaseException as error:
            self._failure('actual_parent_sensor_cycle',error,attempted);raise
        finally:self._in_step=False
    def record_query(self,parent_observation,*,query_index,env_step_index,absolute_env_step_index,query_time_actual):
        try:
            S.need(not self.rec.closed and not self._in_step,'query_outside_actual_step')
            self.issued.require_query_ready(query_index)
            row=self.rec.rows[self.rec.next]if self.rec.next<len(self.rec.rows)else None
            result=None
            if row is not None and query_index>=row['planned_query_index']:
                S.need(query_index==row['planned_query_index'],'frozen_query_skipped')
                def supplied():
                    S.need(self.pending is not None and self.pending['window_slot']==self.rec.next and self.pending['env_step_index']==env_step_index,'arm_before_sensor_cycle_not_at_query')
                    S.need(S.R._identity(self.bindings['root.observation'].snapshot())==S.observation_identity(parent_observation),'actual_parent_consumed_observation')
                    S.need(all(self.tap.records.get(c)is self.pending['records'][c]for c in CAMERAS),'actual_sensor_records_not_replaced')
                    proof=self.sensor_gate(parent_observation,self.tap.records)
                    S.need(proof.get('accepted')is True,'observable_provenance_not_qualified')
                    self._emit('SENSOR_PROVENANCE',{'query_index':query_index,**proof},{})
                    return self.tap.supplier(parent_observation)()
                result=self.rec.record(window_slot=self.rec.next,query_index=query_index,env_step_index=env_step_index,
                    absolute_env_step_index=absolute_env_step_index,query_time_actual=query_time_actual,state=self.state,pair_supplier=supplied)
                self.pending=None
            else:S.need(self.pending is None,'unconsumed_sensor_pair_before_early_query')
            self.issued.authorized_query=query_index
            return result
        except BaseException as error:
            if not self.rec.closed:self._failure('actual_query_pair',error,self.rec.next if self.rec.next<len(self.rec.rows)and query_index>=self.rec.rows[self.rec.next]['planned_query_index']else None)
            raise

class Collector:
    """Passive callbacks around the source producer; no policy clock invented."""
    def __init__(self,plan,state,tap,cursor,clock,emit,*,sensor_gate,diagnose,step_snapshot,charge,storage=None,sensor_return=None):
        S.need(all(callable(f)for f in (emit,sensor_gate,diagnose,step_snapshot,charge)),"explicit_collector_services")
        self.plan=copy.deepcopy(plan);self.state=state;self.tap=tap;self.cursor=cursor;self.clock=clock
        self.emit,self.diagnose,self.step_snapshot=emit,diagnose,step_snapshot
        self.charge=charge;self.storage=storage;self.sensor_return=sensor_return
        S.need(callable(sensor_return)or state.synthetic,"native_returned_observation_event_binding_required")
        S.need(storage is not None or state.synthetic,"native_streaming_query_storage_required")
        self.rec=ScheduledRecorder(plan,emit);self.bridge=ScheduledBridge(self.rec,state,tap,cursor,sensor_gate)
        self.pending_query=None;self.step_trace=[];self.failure=None
    def sync(self,*,query,steps,absolute):
        # Values are supplied by the original producer's actual counters.
        self.clock.update(query_index=query,env_step_index=steps,absolute_env_step_index=absolute,
                          query_time_actual=copy.deepcopy(self.tap.clock_state()['sim_time']))
    def returned(self,obs,*,steps,absolute):
        if self.sensor_return is not None:
            self.sensor_return(obs,env_step_index=steps,absolute_env_step_index=absolute,
                return_time_actual=copy.deepcopy(self.tap.clock_state()["sim_time"]))
    def query_input(self,obs,*,query,steps):
        if self.storage is not None:self.storage.begin_query(query)
        self.sync(query=query,steps=steps,absolute=10+steps)
        paired=self.bridge.record_query(obs,**self.clock)
        self.pending_query=None if paired is None else {'root_trajectory_id':self.plan['root_trajectory_id'],'query_index':query,'pair':paired[0],'proof':paired[1],
            'tier':copy.deepcopy(self.plan['tiers'][str(query)])}
    def charge_parent(self,*,query,noise_label):
        self.charge({"root_trajectory_id":self.plan["root_trajectory_id"],"query_index":query,"condition":"parent_clean",
            "noise_label":noise_label,"noise_role":"parent_actual","kind":"parent_stock","full":1,"suffix":10,"prefill":1})
    def parent_generated(self,generated,noise,*,query):
        self.emit('PARENT_ACTIONS',{'query_index':query,'parent_action_source':'original_stock','proposed_first5_only':True},
                  {'raw32':generated['raw_actions'],'adapter7':generated['actions'],'noise':noise})
        if self.pending_query is not None:
            pending=self.pending_query
            S.need(pending['query_index']==query,'diagnostic_same_actual_query')
            row=self.rec.rows[self.rec.next-1];row['neural_status']='attempted'
            try:
                protected=[S.passive_witness_binding(b.name,b.snapshot)if b.allow_temporary_change else b for b in self.state.bindings]
                with S.R.ReferenceTransaction(protected,required_states=S.DIAGNOSTIC_STATES,
                        physical_step_counter=self.state.step_attempts,generators=self.state.generators) as transaction:
                    result=self.diagnose(pending,generated,noise)
                row['neural_status']='recorded'
                self.emit('DIAGNOSTIC_FINISHED',{'query_index':query,'result':result,'protected_state':transaction.report},{})
            except BaseException as error:
                row['neural_status']='failed'
                self.failure={'phase':'natural_recording','query_index':query,'type':type(error).__name__}
                self.emit('DIAGNOSTIC_FAILED',self.failure,{})
                raise
        self.cursor.bind_generated(generated,query_index=query);self.pending_query=None
    @contextmanager
    def policy_step(self,*,slot,action,query,steps):
        before=copy.deepcopy(self.step_snapshot())
        with self.bridge.producing_step(next_n=steps+1,next_absolute=11+steps,slot=slot,action=action):
            yield
        after=copy.deepcopy(self.step_snapshot())
        self.emit('ACTUAL_ISSUED_STEP',{'query_index':query,'action_position':slot,'policy_step':steps+1,
            'absolute_env_step_index':11+steps,'issued':True}, {'action':action,'before':before,'after':after})
    def outcome(self,*,query,slot,steps,success,done):
        row={'query_index':query,'action_position':slot,'policy_step':steps,'official_success':success,'done':done}
        self.step_trace.append(row);self.emit('ACTUAL_STEP_OUTCOME',row,{})
    def query_done(self,query):
        if self.storage is not None:self.storage.end_query(query)
    def finish(self,result):
        if self.storage is not None and self.storage.active_query is not None:self.storage.end_query(self.storage.active_query)
        reason='root_terminal'if result['status'].startswith('accepted_')else'root_failure'
        if not self.rec.closed:self.bridge.finish(reason)
        self.emit('ROOT_CLOSED',{'result':result,'planned':self.rec.summary(),
            'unissued_action_cursor':self.cursor.snapshot(),'diagnostic_failure':self.failure},{})
        return self.rec.summary()
