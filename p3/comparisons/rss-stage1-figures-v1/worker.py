"""Read only exact new OUTCOMEs and frozen baseline caches on cloud CPU.

No provider API, model/array IO, adoption, registry write or automatic publish.
Original frozen normalizer and pair gate are called for every newly affected
pair. Prior 3626 tables and old outcomes remain immutable references.
"""

import argparse
import base64
from collections import Counter
import gzip
import importlib.util
import io
import json
from pathlib import Path
import platform
import time


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


PARENT_RESULT_SHA = "5938d81bbf9fa746c2944e84aca6d150dcb7dade9c5e1b4d3576f770f18ef6a7"
PARENT_TABLE_SHAS = {
    "870059244895a2ed6f716cd48b11340de8945caf2ad1cb2d052b719e039909bd",
    "aa37c54aeec3dfc3264922afeeabd772c96e8d972d07a7a4f678c21eb97cf8a7",
}


def bounded_gzip(raw, limit):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        value = stream.read(limit + 1)
    need(len(value) <= limit, "bounded_control_gzip")
    return value


def load_borrowed(path, ref):
    p = Path(path)
    spec = importlib.util.spec_from_file_location("_paper3_borrowed_increment", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(module.file_identity(p) == {k: ref[k] for k in ("bytes", "sha256")}, "borrowed_worker_SHA")
    return module


class SummaryOverlay:
    """Locally bundled exact summaries; only original OUTCOME uses GET."""

    def __init__(self, body, rows, transport, borrowed):
        raw = json.loads(bounded_gzip(body, 8 * 1024**2))
        self.refs = {r["summary"]["uri"]: r["summary"] for r in rows}
        need(
            len(self.refs) == len(rows) == len(raw) and set(raw) == set(self.refs), "complete_unique_packaged_summaries"
        )
        self.summaries = {}
        for uri, encoded in raw.items():
            data = base64.b64decode(encoded, validate=True)
            self.summaries[uri] = borrowed.exact(data, self.refs[uri])
        self.transport, self.seen = transport, set()

    def get(self, uri, limit):
        if uri not in self.summaries:
            return self.transport.get(uri, limit)
        need(uri not in self.seen and limit == self.refs[uri]["bytes"], "one_exact_local_summary_read")
        self.seen.add(uri)
        return self.summaries[uri]


def run(inputs,dependency_root,borrowed_worker_path,store,work,output,deadline_epoch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    need(platform.system()=='Linux' and 0<deadline_epoch-time.time()<=3600,'cloud_render_only');inputs=Path(inputs);task=json.loads((inputs/'INPUT-SPEC-v1.json').read_bytes());need(task['analysis_kind']=='fixed_RSS_stage1_figures'and task['new_count']==0,'plot_existing_data_only')
    borrowed=load_borrowed(borrowed_worker_path,task['borrowed_worker']);need(borrowed.file_identity(__file__)==task['worker_source'],'actual_plot_source');out=borrowed.OutputBudget(output);transport=borrowed.ExactReads(store,task['readset']['objects'],deadline_epoch)
    source=json.loads(transport.get(task['result_source']['uri'],task['result_source']['bytes']));raw=transport.get(task['trace_source']['uri'],task['trace_source']['bytes']);rows=[json.loads(l)for l in bounded_gzip(raw,32*1024**2).splitlines()];need(len(rows)==10530 and source['new_count']==423,'accepted423_trace_scope')
    panel=task['panel'];cases=dict(zip(panel['ordinals'],panel['case_definitions']));groups=defaultdict(Counter)
    for r in rows:
        if r['contrast']not in ['floor_s0_T30_minus_T20','floor_s1_T30_minus_T20']or not r['control_source']or not r['treatment_source']:continue
        groups[(0 if r['contrast'].startswith('floor_s0')else 1,cases[r['ordinal']]['suite'])][r['diagnostic']['reason']]+=1
    direct=[]
    for seed in [0,1]:direct.append({r['ordinal']:r for r in rows if r['contrast']==f'reheat_minus_floor_s{seed}_T30'and r['cell']})
    common=set(direct[0])&set(direct[1]);need(len(common)==102,'same102_matched_direct_support');cells=[Counter(v[o]['cell']for o in common)for v in direct]
    need(cells[0]['FS']==6 and cells[0]['SF']==6 and cells[1]['FS']==5 and cells[1]['SF']==7,'exact_accepted_counts')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12.8,6.8),layout='constrained');grid=fig.add_gridspec(2,3,width_ratios=[1.6,1,1],height_ratios=[1,.28]);ax=fig.add_subplot(grid[0,0]);blue='#326A95';orange='#D47E43'
    suites=['libero_spatial','libero_object','libero_goal','libero_10'];labels=[];passed=[];rejected=[]
    for suite in suites:
        for seed in [0,1]:
            g=groups[(seed,suite)];need(set(g)<={'qualified_recorded_pair','q0_input_difference'},'only_actual_qualification_reasons');labels.append(suite.replace('libero_','')+f' / seed {seed}');passed.append(g['qualified_recorded_pair']);rejected.append(g['q0_input_difference'])
    y=np.arange(len(labels));ax.barh(y,passed,color=blue,label='Qualified');ax.barh(y,rejected,left=passed,color=orange,label='q0 differs');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,40);ax.set_xlabel('Cases with both endpoint records');ax.set_title('A  Parent comparison loses input matching',loc='left',fontweight='bold',pad=14)
    for i,(q,r)in enumerate(zip(passed,rejected)):
        if q:ax.text(q/2,i,str(q),ha='center',va='center',color='white',fontsize=9)
        if r:ax.text(q+r/2,i,str(r),ha='center',va='center',color='white',fontsize=9)
        ax.text(q+r+.5,i,f'n={q+r}',va='center',fontsize=8)
    ax.legend(loc='lower right',bbox_to_anchor=(1,1.01),frameon=False,ncol=2,fontsize=9)
    for seed in [0,1]:
        a=fig.add_subplot(grid[0,seed+1]);v=cells[seed];arr=np.array([[v['SS'],v['SF']],[v['FS'],v['FF']]])
        a.imshow(arr,cmap='Blues',vmin=0,vmax=80);a.set_xticks([0,1],['Success','Failure']);a.set_yticks([0,1],['Success','Failure']);a.set_xlabel('Reheat outcome');a.set_ylabel('Floor outcome');a.set_title(f'{"B"if seed==0 else"C"}  Same 102 cases / seed {seed}',loc='left',fontweight='bold',pad=14)
        names=[['Both succeed','Harm'],['Gain','Both fail']]
        for i in range(2):
            for j in range(2):a.text(j,i,f'{arr[i,j]}\n{names[i][j]}',ha='center',va='center',color='white'if arr[i,j]>40 else'#172A38',fontsize=11)
        a.text(.5,-.18,f'Gain − harm = {v["FS"]-v["SF"]:+d}',ha='center',transform=a.transAxes,fontweight='bold')
    note=fig.add_subplot(grid[1,:]);note.axis('off');note.text(0,.9,'Panel A: floor-T30 vs its same-seed T20 parent; only pairs with both records shown.\nPanels B–C: reheat vs floor, the identical 102 cases qualifying separately in both seeds.\nq0 rejection is not policy failure. These are first-stage partial-support results, not full-panel success rates or confidence intervals.',va='top',fontsize=10,color='#394852')
    fig.suptitle('Training benefit depends on comparable inputs and paired outcomes',fontsize=15,fontweight='bold')
    fig.savefig(out.directory/'STAGE1-EVIDENCE.png',dpi=180);fig.savefig(out.directory/'STAGE1-EVIDENCE.svg');plt.close(fig)
    data={'parent_qualification':[{ 'seed':k[0],'suite':k[1],**dict(v)}for k,v in sorted(groups.items())],'common_direct_cases':sorted(common),'direct_cells':[dict(c)for c in cells],'no_full_population_claim':True}
    with out.gzip('NEW-PAIR-TRACE-v1.jsonl.gz')as stream:stream.write(raw if False else b''.join(borrowed.canonical(r)+b'\n'for r in rows))
    with out.gzip('NORMALIZED-NEW-v1.jsonl.gz')as stream:stream.write(borrowed.canonical(data)+b'\n')
    result={'schema':'paper3-added-endpoint-pairs-v1','analysis_kind':task['analysis_kind'],'model_id':task['model_id'],'run_id':task['run_id'],'new_count':0,'new_endpoint_identity':[],'new_trace_rows':10530,'new_OUTCOME_GETs':0,'old_OUTCOME_GETs':0,'GPU':0,'model_calls':0,'full_population_evaluation_complete':False,'CI':None,'input_objects':2,'input_bytes':transport.bytes,'input_spec':borrowed.file_identity(inputs/'INPUT-SPEC-v1.json'),'figures':[{'relative':n,**borrowed.file_identity(out.directory/n)}for n in ['STAGE1-EVIDENCE.png','STAGE1-EVIDENCE.svg']],'plot_data':data,'render_location':'cloudCPU','source_result':task['result_source'],'source_trace':task['trace_source']}
    out.json('INPUT-RECEIPT-v1.json',{'verified_objects':2,'verified_bytes':transport.bytes,'new_OUTCOME_GETs':0,'old_OUTCOME_GETs':0,'input_spec':result['input_spec']});out.json('RESULT-v1.json',result,reserve=False);return result
