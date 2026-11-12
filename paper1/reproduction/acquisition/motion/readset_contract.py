"""Outcome-blind 536-identity contract for a later, root-frozen actual readset."""
from __future__ import annotations
import csv
import hashlib
import io

SELECTION_SHA='7c79466c43707d9f2f201238b9b763dc626d5beb3efab2a9f438fc1a36bd8f9a'
CASE_PLAN_SHA='fb40fd7922fe37ed14c3a809de0a24db7596bf1a2fa84723a35f1de943beb9fb'
SOURCE_ROLES={
    'v3':('parallel-collection-runtime-v3','8512e29be044ffc7fd5be2e266ac58ffad9b609b0c46e9d7a9c578caf9d69d39'),
    'v4':('parallel-collection-runtime-v6','0a71ea3bf1d5d6d63c5ff328f2f5b50431723765a6c65a7ada193e6ec38db15f'),
    'v5':('parallel-collection-runtime-v6','4bf734cc8add254df61bcb39e9b300e65474fcf53ae5395b19eaafff1fd43b24'),
}
FAMILIES={'Background Textures','Camera Viewpoints','Light Conditions','Sensor Noise'}
TRAJECTORIES={'M/reference/T-trajectory.npz','M/treatment/T-trajectory.npz'}

def require(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()

def validate(readset,case_plan_raw):
    require(readset.get('schema')=='plus-shared-channel-actual536-readset-v1'
            and readset.get('selection_sha256')==SELECTION_SHA
            and readset.get('case_plan_sha256')==CASE_PLAN_SHA
            and sha(case_plan_raw)==CASE_PLAN_SHA,'frozen_536_identity_sources')
    plan=list(csv.DictReader(io.StringIO(case_plan_raw.decode())))
    require(len(plan)==536 and len({r['case_id']for r in plan})==536,'exact_536_planned_identity')
    by_plan={r['case_id']:r for r in plan}
    rows=readset.get('cases')
    require(isinstance(rows,list)and len(rows)==536 and len({r['case_id']for r in rows})==536
            and {r['case_id']for r in rows}==set(by_plan),'all_536_actual_or_missing_rows')
    eligible=[];prefixes=set()
    for row in rows:
        plan_row=by_plan[row['case_id']]
        for key in ('family','suite','noise_label','row_id','parent_task_candidate','cohort','run_id'):
            require(str(row.get(key))==plan_row[key],'actual_row_matches_original_plan_'+key)
        require(row['family']in FAMILIES and row['cohort']in SOURCE_ROLES
                and (row['runtime_source_role'],row['source_package_sha256'])==SOURCE_ROLES[row['cohort']],
                'frozen_v3_v6_source_role')
        status=row.get('M_status')
        require(status in ('complete','technical_missing','blocked','failed','not_started','unavailable'),
                'explicit_actual_M_availability')
        readiness=row.get('T_readiness')
        require(readiness in ('readable','not_collected','archive_missing','member_missing','schema_unavailable','technical_pending'),
                'explicit_T_member_readiness')
        if status!='complete' or readiness!='readable':
            require(readiness!='readable' or status=='complete','no_readable_T_without_complete_M')
            require(not row.get('trajectory_members'),'no_trajectory_claim_for_unavailable')
            continue
        require(row.get('pair_qualified') is True and row.get('T_C_final_exact') is True,
                'actual_pair_and_T_C_qualification')
        require(isinstance(row.get('clean_id'),str)and len(row['clean_id'])>=16
                and isinstance(row.get('task_id'),str)and len(row['task_id'])>=16,
                'actual_clean_and_task_identity')
        scope=row.get('case_scope')
        require(isinstance(scope,str)and scope.startswith('redacted-90e00ae5ddb1')
                and scope.endswith('/') and row['run_id']+'/'in scope and 'cases/case-'in scope,
                'exact_owned_case_scope')
        for key in ('index_uri','manifest_uri'):
            require(isinstance(row.get(key),str)and row[key].startswith(scope),'exact_case_metadata_URI')
        for key in ('index_sha256','manifest_sha256'):
            require(isinstance(row.get(key),str)and len(row[key])==64,'exact_case_metadata_SHA')
        members=row.get('trajectory_members')
        require(isinstance(members,list)and {r['member']for r in members}==TRAJECTORIES
                and len(members)==2,'two_actual_T_native_members_only')
        for member in members:
            require(0<int(member['bytes'])<=64*1024*1024 and len(member['sha256'])==64,
                    'bounded_original_T_member_identity')
        eligible.append(row);prefixes.add(scope.split('cases/case-')[0])
    require(len(prefixes)<=64,'bounded_exact_read_scopes')
    return {'planned':536,'eligible':len(eligible),'unavailable':536-len(eligible),
            'eligible_rows':eligible,'read_prefixes':sorted(prefixes),
            'source_role_counts':{cohort:sum(r['cohort']==cohort for r in eligible)for cohort in SOURCE_ROLES}}
