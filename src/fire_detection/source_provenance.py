"""Verify a detached release against a real Git repository and byte snapshots.

HEAD records ancestry only. Execution identity is the complete hashed snapshot.
This module uses only the Python standard library and never initializes CUDA.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_SERENA_STATUS = frozenset({b'?? .serena/.gitignore', b'?? .serena/project.yml'})
_SERENA_FILES = frozenset({'.gitignore', 'project.yml', 'project.local.yml'})


def _safe_serena_metadata(repository, expected_semantics_sha256):
    """Permit only inert, reviewed Serena metadata; scan Git-ignored files too."""
    directory = repository / '.serena'
    if not directory.is_dir():
        raise ValueError('Serena metadata directory missing')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if not {'.gitignore', 'project.yml'} <= actual or not actual <= _SERENA_FILES:
        raise ValueError(f'Unexpected Serena metadata file: {sorted(actual - _SERENA_FILES)}')
    if (directory / '.gitignore').read_text(encoding='utf-8').splitlines() != ['/cache', '/project.local.yml']:
        raise ValueError('Serena ignore rules changed')
    import yaml
    project = yaml.safe_load((directory / 'project.yml').read_text(encoding='utf-8'))
    if not isinstance(project, dict) or project.get('project_name') != repository.name:
        raise ValueError('Serena project identity changed')
    canonical_semantics = hashlib.sha256(json.dumps(project, sort_keys=True,
        ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if canonical_semantics != expected_semantics_sha256:
        raise ValueError('Unreviewed Serena project configuration semantics changed')
    inactive = {
        'activation_command': (None, ''),
        'initial_prompt': (None, ''),
        'languages': (None, []),
        'ls_additional_workspace_folders': (None, []),
        'ignored_paths': (None, []),
        'excluded_tools': (None, []),
        'included_optional_tools': (None, []),
        'fixed_tools': (None, []),
        'default_modes': (None, []),
        'added_modes': (None, []),
        'read_only_memory_patterns': (None, []),
        'ignored_memory_patterns': (None, []),
    }
    for key, acceptable in inactive.items():
        if project.get(key) not in acceptable:
            raise ValueError(f'Serena setting may affect execution: {key}')
    if 'project.local.yml' in actual:
        overrides = yaml.safe_load((directory / 'project.local.yml').read_text(encoding='utf-8'))
        if overrides not in (None, {}):
            raise ValueError('Serena local overrides are not inert')
    return {name: file_sha(directory / name) for name in sorted(actual)}

def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def _checked_file(root, name, expected):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or file_sha(path)!=expected:
        raise ValueError(f'Source provenance file drift: {name}')

def verify_execution_source(root):
    root=Path(root).resolve()
    path=root/'provenance/manifest.json'
    record=json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema_version')!=2 or record.get('head_proves_execution_bytes') is not False:
        raise ValueError('Unsupported source provenance contract')
    repository=Path(record['repository']).resolve()
    if not (repository/'.git').exists():
        raise ValueError('Real source Git metadata missing')
    def git(*args):
        result=subprocess.run(['git','--no-optional-locks','-C',str(repository),*args],capture_output=True)
        if result.returncode:
            raise ValueError(f'Git provenance command failed: {args}; exit={result.returncode}; stderr={result.stderr.decode("utf-8",errors="replace")}')
        return result.stdout
    top=Path(git('rev-parse','--show-toplevel').decode().strip()).resolve()
    head=git('rev-parse','HEAD').decode().strip()
    status=git('status','--porcelain=v1','--untracked-files=all')
    if top!=repository or head!=record['head']:
        raise ValueError('Real repository path or HEAD drift')
    status_lines = status.splitlines()
    permitted = [line for line in status_lines if line in _SERENA_STATUS]
    core_status = b'\n'.join(line for line in status_lines if line not in _SERENA_STATUS) + b'\n'
    if set(permitted) != _SERENA_STATUS or hashlib.sha256(core_status).hexdigest() != record['git_status_core_sha256']:
        raise ValueError('Real repository protected Git state drift')
    metadata_hashes = _safe_serena_metadata(repository, record['serena_project_semantics_sha256'])
    for name,expected in record['execution_files'].items():
        _checked_file(root,name,expected)
        _checked_file(root,'provenance/execution_snapshot/'+name,expected)
    actual_python={p.relative_to(root).as_posix() for d in ['src','scripts'] for p in (root/d).rglob('*.py')}
    if actual_python!={n for n in record['execution_files'] if n.endswith('.py')}:
        raise ValueError('Unexpected or missing executable Python source')
    for name,expected in record['upstream_working_files'].items():
        _checked_file(repository,name,expected)
        _checked_file(root,'provenance/upstream_working_snapshot/'+name,expected)
    for name,expected in record['evidence_files'].items():
        _checked_file(root,name,expected)
    audit = root / 'runs/provenance_metadata_audit.jsonl'
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),
            permitted_status=[line.decode('utf-8') for line in permitted],
            metadata_sha256=metadata_hashes, core_status_sha256=hashlib.sha256(core_status).hexdigest()),
            sort_keys=True) + '\n')
    return dict(commit=head,dirty=bool(status.strip()),repository=str(repository),
        dirty_status_sha256=hashlib.sha256(status).hexdigest(),
        protected_status_sha256=hashlib.sha256(core_status).hexdigest(),
        permitted_metadata_sha256=metadata_hashes,
        execution_snapshot_sha256=hashlib.sha256(json.dumps(record['execution_files'],sort_keys=True).encode()).hexdigest(),
        provenance_manifest_sha256=file_sha(path),release_id=record['release_id'],
        head_proves_execution_bytes=False,source_verification='REAL_GIT_PLUS_FROZEN_EXECUTION_AND_UPSTREAM_SNAPSHOTS')
