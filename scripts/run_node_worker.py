"""Internal node child; repeats release-bound authorization before execution."""
from __future__ import annotations
import argparse
import importlib.util
import json
import multiprocessing
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runs/.ultralytics')
os.environ['PYTHONDONTWRITEBYTECODE']='1'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--approval',type=Path,required=True)
    args=parser.parse_args()
    from fire_detection.research.three_arm_recovery import NodeEngine,read,atomic
    engine=NodeEngine(ROOT,args.plan.parent)
    spec=importlib.util.spec_from_file_location('authorized_runner',ROOT/'scripts/run_sealed_three_arm_v2.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    approval=read(args.approval)
    runner.require_authorization(engine,approval['scope'],args.approval)
    if args.task not in approval['scope']:
        raise PermissionError('Child task outside scope')
    task=engine.tasks[args.task]
    attempt=args.attempt.resolve()
    if attempt.parent != (engine.target(task)/'execution').resolve():
        raise PermissionError('Child attempt escapes selected task')
    record=read(attempt/'record.json')
    if record['task']!=args.task or record['status']!='RUNNING':
        raise PermissionError('Child requires a live reserved parent attempt')
    from fire_detection.research.node_logging import bound_ultralytics_logger
    from fire_detection.research.immutable_inputs import frozen_inputs
    from fire_detection.research.protocol import load_config
    cfg=load_config(ROOT/engine.plan['base_config'],root=ROOT)
    with bound_ultralytics_logger() as logger, frozen_inputs(cfg) as inputs:
        atomic(attempt/'input_snapshot_receipt.json',inputs)
        from fire_detection.research.three_arm_execution import execute_node
        logger.info('NODE_BEGIN task=%s',args.task)
        paths=execute_node(ROOT,engine.release,engine.plan,task,attempt)
        logger.info('NODE_COMPLETE task=%s',args.task)
    # Publish completion only after frozen_inputs performs its final validation.
    atomic(attempt/'child_products.json',dict(task=args.task,
           paths=[str(Path(p).resolve()) for p in paths]+[str(attempt/'input_snapshot_receipt.json')]))
    return 0

if __name__=='__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
