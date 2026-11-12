"""Cloud-only old-record numerical readout and Matplotlib rendering. No native execution."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
import traceback
import physical_codec as C
import readout as R

HERE = Path(__file__).resolve().parent
RESULT = 'q11-physical-figures-result.json'
OUTS = [RESULT, 'FIGURE-VALUES.json', 'PHYSICAL-PAIRS.csv', 'QA.json', 'CAPTIONS.json',
        'PROTOCOL.json', 'fig_physical_response.pdf', 'fig_physical_response.svg', 'fig_physical_response.png',
        'fig_physical_support.pdf', 'fig_physical_support.svg', 'fig_physical_support.png']


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def run(stage, out, manifest_sha, deadline):
    C.need(sys.platform.startswith('linux') and math.isfinite(deadline), 'cloud_Linux_actual_readout')
    stage, out = Path(stage), Path(out)
    C.need(not out.exists(), 'new_figure_output_directory')
    out.mkdir(parents=True)
    result = {'schema':'paper2-q11-physical-figures-result-v1','status':'FAILED_PRESERVED',
        'analysis_complete':False,'input_qualified':False,'synthetic':False,'GPU_count':0,
        'new_model_calls':0,'new_env_steps':0,'new_fits':0,'training_updates':0,'JPFS_new_bytes':0,
        'runtime_imports':0,'environment_constructions':0}
    def check():
        C.need(time.time()<deadline,'figure_science_absolute_deadline')
    def write(name, value):
        raw=encoded(value)
        C.need(len(raw)<=2*1024**2,'small_figure_metadata')
        with (out/name).open('xb')as f:f.write(raw)
    started=time.monotonic();phase='source'
    try:
        check();raw=(HERE/'SOURCE-MANIFEST.json').read_bytes()
        C.need(C.identity(raw)['sha256']==manifest_sha,'exact_figure_source_manifest')
        manifest=C.json_bytes(raw)
        C.need(manifest['schema']=='paper2-q11-physical-figures-source-v1','exact_source_schema')
        for ref in manifest['files']:
            p=HERE/ref['path'];C.need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(HERE) and
                C.identity(p.read_bytes())=={k:ref[k]for k in('bytes','sha256')},'exact_figure_source_member')
        ir=(HERE/'INPUTS.json').read_bytes();inputs=C.json_bytes(ir)
        result.update(source_manifest=C.identity(raw),input_manifest=C.identity(ir))
        refs={r['path']:r for r in inputs['input_objects']}
        C.need(len(refs)==inputs['input_object_count']==13 and
               sum(r['bytes']for r in refs.values())==inputs['input_body_bytes']==424142,'fixed13_existing_records')
        C.need({p.name for p in stage.iterdir()}==set(refs),'exact_figure_stage_roster')
        phase='payload_and_provenance';docs={};decoded_arrays=0
        for name,ref in refs.items():
            check();p=stage/name
            C.need(p.is_file() and not p.is_symlink() and p.stat().st_size==ref['bytes'] and
                   C.identity(p.read_bytes())=={k:ref[k]for k in('bytes','sha256')},'figure_input_whole_SHA')
            if name.endswith('.gz'):
                docs[ref['logical_path']],n=C.load(p);decoded_arrays+=n['arrays']
            else:
                counts={'arrays':0,'numeric_bytes':0}
                docs[ref['logical_path']]=C.decode(C.json_bytes(p.read_bytes()),counts)
                decoded_arrays+=counts['arrays']
        by=R.validate(docs,inputs,HERE)
        rows=R.compute(by);R.compare_original(rows,docs['new/science/PHYSICAL-CONTRASTS.json'])
        phase='all_observed_figures'
        from render import render
        render(rows,out)
        unexpected=[n for n in sys.modules if n.split('.')[0]in('mujoco','robosuite','libero','openpi','jax','jaxlib','torch')]
        result['runtime_imports']=len(unexpected);C.need(not unexpected,'no_target_runtime_imports')
        check()
        proof={'schema':'paper2-q11-physical-figure-values-v1','root_id':inputs['root_id'],'query_index':11,
            'independent_roots':1,'states':1,'technical_N_repeat':True,'timepoints_per_pair':5,
            'full_trajectory_hashes_recomputed':4,'old_N_status':'FAILED_NO_RETRY','pairs':rows,
            'all_nonterminal':True,'task_success_rate_estimated':False,'posthoc_thresholds':False}
        write('FIGURE-VALUES.json',proof)
        flat=[]
        for row in rows:
            item={k:v for k,v in row.items()if k not in('eef_delta_xyz_m','gripper_qpos_delta_original_units')}
            item.update({'eef_delta_'+axis+'_m':row['eef_delta_xyz_m'][i]for i,axis in enumerate('xyz')})
            item.update({'finger'+str(i+1)+'_qpos_delta_original_units':row['gripper_qpos_delta_original_units'][i]for i in range(2)})
            flat.append(item)
        with(out/'PHYSICAL-PAIRS.csv').open('x',newline='')as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        write('QA.json',{'pass':True,'all15_pair_time_rows':True,'original10_pairs_recomputed_exact':True,
            'same_q11_starts_and_exact_N_repeat':True,'typed_arrays_verified':decoded_arrays,
            'all5timepoints_retained':True,'S_fourth_position_dip_retained':True,
            'R_last_contact_count_difference_retained':True,'equal_gripper_targets_not_equal_finger_states':True,
            'same_sample_threecontrasts_not_independent_samples':True,
            'original_N_close_failure_kept':True,'no_task_success_or_generalization_claim':True})
        write('CAPTIONS.json',{
            'main':'Five-command physical responses at one DEV state (n=1 root/state). The N repeat exactly matches the retained N trajectory; all starts match. S-N and R-N position and orientation distances, signed finger joint-state differences, and N/S/R contact counts are shown at all five times. Direct gripper targets match, but finger states need not. The R final contact count differs by one; counts alone do not identify objects or task benefit. Lines connect observed samples and are not fitted dynamics.',
            'support':'All three pairwise contrasts S-N, R-N and S-R over all five times, including signed xyz displacement, both finger states and contact-count differences. These 15 rows are dependent comparisons of one state; no confidence interval or success-rate estimate is claimed.',
            'units':'Position m converted to micrometres; orientation rad converted to microradians. Finger qpos retains simulator units, scaled by10^-6 for display.'})
        write('PROTOCOL.json',C.json_bytes((HERE/'PROTOCOL.json').read_bytes()))
        files=[p for p in out.iterdir()if p.is_file()];C.need(sum(p.stat().st_size for p in files)<16*1024**2,'all_figure_outputs_bounded')
        result.update(status='Q11_PHYSICAL_FIGURES_COMPLETE',analysis_complete=True,input_qualified=True,
            input_objects_verified=13,input_body_bytes=424142,counts={'physical_pairs':3,'timepoints_per_pair':5,
            'pair_time_rows':15,'original_comparisons_exact':10,'trajectory_hashes_recomputed':4,'figures':2},
            typed_arrays_verified=decoded_arrays,output_files=[{'path':p.name,**C.identity(p.read_bytes())}for p in files])
    except Exception as error:
        result['failure']={'phase':phase,'type':type(error).__name__,'code':str(error)[:180],
            'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}for f in traceback.extract_tb(error.__traceback__)[-8:]]}
    result['elapsed_seconds']=time.monotonic()-started
    write(RESULT,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--deadline-epoch',type=float,required=True)
    a=p.parse_args();r=run(a.stage,a.out,a.source_manifest_sha256,a.deadline_epoch)
    print(json.dumps({'status':r['status'],'analysis_complete':r['analysis_complete']}))
    raise SystemExit(0 if r['analysis_complete'] else 2)
