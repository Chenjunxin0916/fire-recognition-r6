"""CPU preflight reached only through the sealed production runner.

Uses the actual task executor and trainer classes, stopping YOLO.train before
gradient computation. All outputs live in a uniquely named engineering campaign.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import traceback
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from .three_arm_recovery import read, sha


class _BeforeOptimization(RuntimeError):
    pass


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _static_matrix(engine):
    from .split_role_guard import training_roles
    rows = []
    partitions = {}
    for outer in range(5):
        for seed in (42, 2026, 3407):
            part = read(engine.release / f'partitions/outer_{outer}/seed_{seed}.json')
            splits = part['splits']
            names = {role: {r['filename'] for r in splits[role]} for role in ('fit', 'cal', 'test')}
            groups = {role: {r['group_id'] for r in splits[role]} for role in names}
            if [len(names[x]) for x in ('fit', 'cal', 'test')] != [660, 220, 220]:
                raise ValueError(f'Partition role sizes differ: outer={outer} seed={seed}')
            if any(names[a] & names[b] or groups[a] & groups[b]
                   for a, b in (('fit', 'cal'), ('fit', 'test'), ('cal', 'test'))):
                raise ValueError(f'Partition role/group leakage: outer={outer} seed={seed}')
            partitions[(outer, seed)] = part
    for task in engine.plan['tasks']:
        if task['kind'] not in ('miner', 'train'):
            continue
        outer, seed = task['outer_fold'], task['seed']
        part = partitions[(outer, seed)]
        fit, cal = training_roles(part, task)
        if task['kind'] == 'miner':
            if len(fit) != 440 or len(cal) != 220 or task['inner_fold'] in (outer, (outer + 1) % 5):
                raise ValueError(f'Miner OOF role drift: {task["id"]}')
            status = 'PASS_IMPORTED_FORMAL_OOF' if task['id'] in engine.imported_ids else 'PASS_CONFIG_DEPENDENCY_PENDING_GPU'
            k, updates, exposures = None, task['expected_optimizer_steps'], task['expected_sample_exposures']
        else:
            if len(fit) != 660 or len(cal) != 220 or task['depends_on'] != [f'select_o{outer}']:
                raise ValueError(f'Classifier FIT/CAL role drift: {task["id"]}')
            selection = engine.completed(f'select_o{outer}')
            if selection is None:
                status, k = 'BLOCKED_BY_DEPENDENCY', None
            else:
                status = 'PASS_CONFIG_ACTUAL_H_READY'
                cell = engine.root / 'runs/direction_a' / engine.plan['imported_completed']['parent_campaign'] / f'outer_{outer}/seed_42'
                k = read(cell / 'hard_negative_selection.json')['K']
            updates = 40 * math.ceil((660 + (k or 0) if task['method'] == 'hn' else 660) / 64) if k is not None or task['method'] != 'hn' else None
            exposures = 40 * (660 + (k or 0) if task['method'] == 'hn' else 660) if k is not None or task['method'] != 'hn' else None
        rows.append(dict(task=task['id'], kind=task['kind'], outer=outer, seed=seed,
                         method=task['method'] or '', inner_fold=task['inner_fold'] if task['kind']=='miner' else '',
                         status=status, fit=len(fit), cal=len(cal), test=len(part['splits']['test']),
                         H_K=k if k is not None else '', expected_updates=updates if updates is not None else '',
                         expected_exposures=exposures if exposures is not None else '',
                         output_dir=task['output_dir'], dependency=','.join(task['depends_on'])))
    if len(rows) != 60 or len({r['output_dir'] for r in rows}) != 60:
        raise ValueError('Classifier/miner task matrix or output isolation changed')
    return rows, partitions


def _probe_method(engine, task, part, hard, run_id, evidence):
    import torch
    from .node_logging import bound_ultralytics_logger
    from .protocol import load_config
    from .three_arm_execution import execute_training
    from .training import configure_runtime

    method = task['method']
    config = load_config(engine.root / engine.plan['base_config'], root=engine.root)
    config['campaign'] = f'focal_cpu_preflight_{run_id}'
    config['engineering_only'] = True
    config['training']['device'] = 'cpu'
    configure_runtime(config)
    from ultralytics import YOLO
    cell = engine.root / 'runs/direction_a' / config['campaign'] / f'outer_{task["outer_fold"]}/seed_{task["seed"]}'
    target = cell / 'models' / method
    attempt = target / 'execution/attempt_001'
    attempt.mkdir(parents=True, exist_ok=False)
    captured = {}

    def stop_before_training(model, trainer, **kwargs):
        if kwargs['device'] != 'cpu' or kwargs['batch'] != 64 or kwargs['nbs'] != 64 or kwargs['epochs'] != 40:
            raise ValueError('Production training arguments changed')
        for key in ('optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'cos_lr', 'amp', 'workers'):
            if kwargs[key] != config['training'][key]:
                raise ValueError(f'Production optimizer/schedule argument changed: {key}')
        actual = trainer(overrides={**kwargs, 'model': str(engine.root / config['classifier']['weights'])})
        actual.device = torch.device('cpu')
        actual.world_size = 1
        actual.data = {'nc': 2}
        actual.model = model.model
        view = Path(kwargs['data'])
        loader = actual.get_dataloader(str(view / 'train'), batch_size=64, rank=-1, mode='train')
        dataset = loader.dataset
        train_names = {r['filename'] for r in part['splits']['fit']}
        if len(dataset) != 660 + (len(hard) if method == 'hn' else 0):
            raise ValueError('Actual production Dataset exposure length differs')
        samples = dataset.samples
        source_names = [Path(str(item[0])).name for item in samples]
        if set(source_names) != train_names or len(source_names) != 660:
            raise ValueError('Actual production Dataset FIT identity differs')
        canonical = engine.root / 'data/processed/dfire_eoi_snapshot_20261020_v1/images'
        if any(not Path(str(item[0])).samefile(canonical / Path(str(item[0])).name) for item in samples):
            raise ValueError('Production view is not the frozen image snapshot')
        if method == 'hn':
            indexed = Counter(source_names[i] for i in dataset.indices)
            expected = Counter(source_names); expected.update(hard)
            if indexed != expected:
                raise ValueError('Actual HN ReplayDataset does not preserve complete H')
        if method == 'focal':
            if actual.focal_state.at_epoch != 10 or actual.focal_state.epochs != 40:
                raise ValueError('Focal uniform/weighted transition differs')
        batch = next(iter(loader))
        names = [Path(str(name)).name for name in batch['im_file']]
        labels = [int(value) for value in batch['cls']]
        by_name = {r['filename']: int(r['label']) for r in part['splits']['fit']}
        if len(names) != 64 or any(name not in by_name or label != by_name[name] for name, label in zip(names, labels)):
            raise ValueError('CPU batch label or sample identity differs')
        if tuple(batch['img'].shape) != (64, 3, 384, 384) or not bool(torch.isfinite(batch['img']).all()):
            raise ValueError('CPU image batch has invalid shape or pixel values')
        optimizer = actual.build_optimizer(model=model.model, name=kwargs['optimizer'],
            lr=kwargs['lr0'], momentum=kwargs['momentum'], decay=kwargs['weight_decay'], iterations=440)
        if optimizer.__class__.__name__ != 'AdamW':
            raise ValueError('Actual optimizer is not AdamW')
        from torch.utils.data import DataLoader
        from .focal_sampling import seed_worker
        worker_loader = DataLoader(dataset, batch_size=8, num_workers=4,
            multiprocessing_context='spawn', worker_init_fn=seed_worker,
            persistent_workers=False)
        worker_batch = next(iter(worker_loader))
        if len(worker_batch['im_file']) != 8:
            raise ValueError('Windows spawn workers returned an incomplete batch')
        del worker_loader
        if hasattr(loader, 'iterator') and hasattr(loader.iterator, '_shutdown_workers'):
            loader.iterator._shutdown_workers()
        captured.update(method=method, seed=task['seed'], dataset_type=type(dataset).__name__,
                        dataloader_type=type(loader).__name__, workers=loader.num_workers,
                        explicit_spawn_workers=4,
                        dataset_size=len(dataset), batch_size=len(names), batch_names=names,
                        batch_labels=labels, optimizer_type=type(optimizer).__name__,
                        epochs=kwargs['epochs'], planned_updates=40 * len(loader),
                        planned_exposures=40 * len(dataset), no_gradient_steps=True,
                        pretrained_weights_sha256=sha(engine.root / config['classifier']['weights']))
        del optimizer, actual, loader
        raise _BeforeOptimization('Expected CPU engineering stop before gradients')

    try:
        with bound_ultralytics_logger(), patch.object(YOLO, 'train', stop_before_training):
            execute_training(config, engine.plan, task, target, cell, part, attempt)
    except _BeforeOptimization:
        pass
    if not captured:
        raise ValueError('Production task did not reach CPU batch and optimizer preflight')
    if method == 'hn':
        if (cell.parent / 'seed_42/hard_negative_selection.json').exists():
            raise ValueError('Preflight HN used a copied test-campaign H')
        budget = read(attempt / 'budget.json')
        if (budget['K'], budget['optimizer_updates'], budget['sample_exposures']) != (41, 440, 28040):
            raise ValueError('Production HN Budget B differs from formal H=41')
    if (target / 'accepted.json').exists() or (target / 'completion.json').exists() or list(target.glob('completed_*.json')):
        raise ValueError('CPU preflight generated a formal completion claim')
    if list(target.rglob('*.pt')):
        raise ValueError('CPU preflight wrote model weights')
    manifest_files = list(target.rglob('research_manifest.json')) + list(target.rglob('focal_manifest.json'))
    if len(manifest_files) != 1 or read(manifest_files[0]).get('status') != 'failed':
        raise ValueError('Expected training failure receipt was not preserved')
    captured['failure_receipt_files'] = [str(p.relative_to(engine.root)) for p in manifest_files]
    captured['status'] = 'PASS'
    _save(evidence / f'{task["id"]}.json', captured)
    return captured


def run_preflight(engine, selected, run_id):
    if not run_id.isascii() or not run_id.replace('_', '').isalnum():
        raise ValueError('Preflight id must contain only ASCII letters, digits and underscore')
    evidence = engine.root / 'runs/preflight' / run_id
    evidence.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        from .protocol import digest_object, load_config, verify_source
        from .immutable_inputs import frozen_inputs
        from .three_arm_execution import runtime_environment
        import yaml
        cfg = load_config(engine.root / engine.plan['base_config'], root=engine.root)
        runtime_policy = yaml.safe_load((engine.root / engine.plan['tasks'][0]['runtime_config']).read_text(encoding='utf-8'))
        runtime_environment(engine.root, runtime_policy, gpu=False)
        _, source = verify_source(cfg)
        if digest_object(source) != engine.plan['source_sha256']:
            raise ValueError('Training source, environment, or data hash changed')
        with frozen_inputs(cfg) as frozen:
            _save(evidence / 'input_snapshot.json', frozen)
        matrix, partitions = _static_matrix(engine)
        with (evidence / 'all_tasks_preflight_matrix.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
            writer.writeheader(); writer.writerows(matrix)
        parent = engine.root / 'runs/direction_a' / engine.plan['imported_completed']['parent_campaign'] / 'outer_0/seed_42'
        receipt = read(parent / 'outer0_remining_receipt.json')
        selection = read(parent / 'hard_negative_selection.json')
        oof = list(csv.DictReader((parent / 'fit_mining_oof.csv').open(encoding='utf-8-sig', newline='')))
        fit = {r['filename']: r for r in partitions[(0,42)]['splits']['fit']}
        if (receipt['status'] != 'FORMAL_OUTER0_OOF_AND_H_ACCEPTED' or len(oof) != 660
            or {r['filename'] for r in oof} != set(fit) or selection['K'] != 41
            or selection['threshold'] != .45 or len(selection['hard_names']) != 41
            or not set(selection['hard_names']) <= {n for n,r in fit.items() if r['label']==0}):
            raise ValueError('Formal Outer 0 OOF/H identity differs')
        for task_id in engine.imported_ids:
            if engine.completed(task_id) is None:
                raise ValueError(f'Formal imported OOF dependency missing: {task_id}')
        selected_tasks = [engine.tasks[task_id] for task_id in selected if task_id in engine.tasks]
        if not selected_tasks or any(task['kind'] != 'train' for task in selected_tasks):
            raise ValueError('Classifier preflight requires one or more formal classifier tasks')
        for task in selected_tasks:
            if task['outer_fold'] != 0:
                results.append(dict(task=task['id'], status='BLOCKED_BY_DEPENDENCY'))
                continue
            try:
                hard = selection['hard_names'] if task['method']=='hn' else []
                proof = _probe_method(engine, task, partitions[(task['outer_fold'],task['seed'])], hard, run_id, evidence)
                results.append(dict(task=task['id'], status='PASS', dataset_size=proof['dataset_size'], workers=proof['workers']))
            except BaseException as exc:
                stack = traceback.format_exc()
                (evidence / f'{task["id"]}_traceback.log').write_text(stack, encoding='utf-8')
                results.append(dict(task=task['id'], status='FAIL', error=str(exc), traceback_file=f'{task["id"]}_traceback.log'))
        statuses = Counter(item['status'] for item in results)
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
        accepted = not statuses.get('FAIL') and statuses.get('PASS', 0) > 0 and statuses.get('PASS', 0) + statuses.get('BLOCKED_BY_DEPENDENCY', 0) == len(results)
        overall = 'PASS_WITH_DEPENDENCY_BLOCKED' if accepted and statuses.get('BLOCKED_BY_DEPENDENCY') else ('PASS' if accepted else 'FAIL')
        report = dict(release_id=engine.plan['release_id'], engine_identity=engine.identity,
                      source_sha256=engine.plan['source_sha256'], formal_H_sha256=sha(parent/'hard_negative_selection.json'),
                      formal_oof_rows=len(oof), formal_H_K=selection['K'], matrix_tasks=len(matrix),
                      selected=selected, results=results, result_counts=dict(statuses), gpu_training_executed=False,
                      optimizer_steps=0, cuda_initialized=False, formal_cal_test_evaluated=False,
                      status=overall)
        _save(evidence / 'report.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if accepted else 1
    except BaseException:
        stack = traceback.format_exc()
        (evidence / 'preflight_traceback.log').write_text(stack, encoding='utf-8')
        _save(evidence / 'report.json', dict(status='FAIL', results=results, traceback_file='preflight_traceback.log',
                                             gpu_training_executed=False, optimizer_steps=0))
        print(stack)
        return 1
