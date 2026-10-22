"""Additional immutable identity checks for the repaired split release."""
import json, csv
from pathlib import Path
from .protocol import digest_object, project_path
from fire_detection.models.offline import sha256_file
def verify_binding(config, records, manifest, image_hashes, label_hashes):
    root=Path(config['_root']);data=config['data'];folder=project_path(root,data['processed_manifest']).parent
    contract=json.loads(project_path(root,data['source_contract']).read_text())
    mapping={'folds.csv':'folds_sha256','groups.csv':'groups_sha256','association_ledger_reviewed.csv':'association_sha256','human_review_history.csv':'author_history_sha256','human_review_decisions.csv':'author_events_sha256','author_binding.json':'author_binding_sha256','retained_edges.csv':'retained_edges_sha256','residual_candidates_final.csv':'residual_candidates_sha256'}
    if sha256_file(folder/'selection_contract.json') != manifest['selection_contract_sha256']:
        raise ValueError('Selection manifest binding drift')
    for name,key in mapping.items():
        if sha256_file(folder/name)!=contract[key]:raise ValueError('Audit/fold identity drift: '+name)
    if contract['dataset_id']!=data['dataset_id'] or contract['dataset_profile']!=data['dataset_profile']:
        raise ValueError('New dataset identity mismatch')
    if len({r.group_id for r in records})!=427 or contract['components']!=427:raise ValueError('Component identity/count drift')
    with (folder/'groups.csv').open(encoding='utf-8-sig',newline='') as f:groups=list(csv.DictReader(f))
    gm={r['filename']:r['group_id'] for r in groups}
    if len(gm)!=len(groups) or gm!={r.filename:r.group_id for r in records}:
        raise ValueError('Group table differs from training records')
    with (folder/'groups.csv').open(encoding='utf-8-sig',newline='') as f:
        group_rows=list(csv.DictReader(f))
    if len(group_rows)!=len(records) or {r['filename']:(r['group_id'],int(r['label'])) for r in group_rows}!={r.filename:(r.group_id,r.label) for r in records}:
        raise ValueError('Component member mapping differs from training records')
    if contract['status']!='FROZEN_RETAINED_ASSOCIATION_GRAPH' or contract['fold_freeze_approved'] is not True:
        raise ValueError('Final retained graph freeze is missing')
    rows={r['filename']:r for r in manifest['images']}
    if len(rows)!=len(manifest['images']):raise ValueError('Duplicate manifest identity')
    for n,h in label_hashes.items():
        if rows[n]['label_sha256']!=h:raise ValueError('Label byte hash drift: '+n)
    return {'contract_sha256':sha256_file(folder/'selection_contract.json'),'author_review_version':contract['author_review_version'],'author_history_sha256':contract['author_history_sha256'],'component_count':427,'group_unit':'conservative_visual_component','source_fingerprint':digest_object({'rows':manifest['images'],'audit':contract})}
