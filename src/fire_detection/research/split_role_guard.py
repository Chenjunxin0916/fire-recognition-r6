"""Pure identity checks before repaired-split training or OOF prediction."""
from .protocol import digest_object
from .three_arm_evaluation import validate_partition

def training_roles(partition, task):
    validate_partition(partition)
    fit, cal = partition['splits']['fit'], partition['splits']['cal']
    if task['kind'] == 'miner':
        inner=task['inner_fold']
        if inner not in partition['fit_folds']:raise ValueError('Miner held fold outside FIT')
        cal=[r for r in fit if r['fold']==inner]
        fit=[r for r in fit if r['fold']!=inner]
        if (len(fit),len(cal))!=(440,220):raise ValueError('Miner role size drift')
    elif task['kind']!='train':raise ValueError('Not a training node')
    elif (len(fit),len(cal))!=(660,220):raise ValueError('Classifier role size drift')
    test={r['filename'] for r in partition['splits']['test']}
    train_names={r['filename'] for r in fit};val_names={r['filename'] for r in cal}
    if test & (train_names|val_names) or train_names & val_names:raise ValueError('Role identity contamination')
    return fit,cal

def require_oof_rows(partition, rows, model_name):
    validate_partition(partition)
    allowed={f"miner_inner_{m['validation_fold']}":m['validation_fold'] for m in partition['miners']}
    if model_name not in allowed:raise ValueError('Unknown FIT miner')
    expected=[r for r in partition['splits']['fit'] if r['fold']==allowed[model_name]]
    if digest_object(rows)!=digest_object(expected):raise ValueError('OOF rows differ from exact FIT held fold')
    return True
