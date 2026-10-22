"""Versioned inputs: write sharing constraints plus identity/hash verification.

Sharing handles constrain ordinary content writers, not every same-user namespace
operation. Deletion/replacement is detected and invalidates the affected run.
No ACL, original file attribute, or source byte is changed by this module.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def deny_writes(paths):
    """Hold read-sharing handles; do not claim universal deletion prevention."""
    if os.name != 'nt':
        raise RuntimeError('This Windows execution contract requires read-sharing handles')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handles = []
    try:
        for path in dict.fromkeys(str(Path(p).resolve()) for p in paths):
            # GENERIC_READ, FILE_SHARE_READ only, OPEN_EXISTING, NORMAL.
            handle = create(path, 0x80000000, 0x1, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            if not close(handle):
                raise ctypes.WinError(ctypes.get_last_error())


def binding(config):
    root = Path(config['_root']).resolve()
    value = config['data'].get('input_snapshot_manifest')
    if not value:
        raise ValueError('Mandatory common input snapshot missing')
    manifest = (root / value).resolve()
    if not manifest.is_relative_to(root):
        raise ValueError('Input snapshot manifest escapes release')
    expected = config['data']['input_snapshot_manifest_sha256']
    if sha(manifest) != expected:
        raise ValueError('Input snapshot manifest drift')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data['status'] != 'FROZEN_COMMON_INPUT_SNAPSHOT' or data['dataset_id'] != config['data']['dataset_id']:
        raise ValueError('Input snapshot identity differs')
    return root, manifest, data


class InputDrift(ValueError):
    """An input changed; no result from the guarded call can be accepted."""


def file_identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def verify_files(files, identities=None):
    for path, expected in files.items():
        try:
            if identities is not None and file_identity(path) != identities[path]:
                raise InputDrift(f'Frozen input identity changed: {path}')
            if sha(path) != expected:
                raise InputDrift(f'Frozen input bytes changed: {path}')
        except OSError as error:
            raise InputDrift(f'Frozen input unavailable: {path}') from error


@contextmanager
def verified_read(path, expected):
    """Validate each actual Dataset/prediction read, including exception exits."""
    path = Path(path)
    with deny_writes([path]):
        identity = {path:file_identity(path)}
        verify_files({path:expected}, identity)
        try:
            yield
        finally:
            verify_files({path:expected}, identity)


@contextmanager
def frozen_inputs(config):
    root, manifest, data = binding(config)
    files = {manifest:config['data']['input_snapshot_manifest_sha256']}
    for name, expected in data['files'].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root):
            raise ValueError('Input snapshot member escapes release')
        files[p] = expected
    with deny_writes(files):
        identities = {p:file_identity(p) for p in files}
        verify_files(files, identities)
        try:
            yield dict(dataset_id=data['dataset_id'],manifest_sha256=sha(manifest),
                       normalization_policy=data['normalization_policy'],
                       write_guard='WIN32_READ_SHARING_PLUS_PER_READ_AND_PRE_POST_IDENTITY_SHA256',
                       absolute_delete_prevention=False,file_count=len(files))
        finally:
            # A failing loader must not skip post-validation. Namespace changes
            # are invalid even if a replacement has the same bytes.
            verify_files(files, identities)


def snapshot_identity(config):
    _, manifest, data = binding(config)
    return dict(dataset_id=data['dataset_id'],manifest_sha256=sha(manifest),
                normalization_policy=data['normalization_policy'])


def guarded_inputs(function):
    @functools.wraps(function)
    def guarded(config, *args, **kwargs):
        import inspect
        arguments = inspect.signature(function).bind(config, *args, **kwargs).arguments
        run_dir = arguments.get('run_dir')
        invalid = None
        if run_dir is not None:
            run_dir = Path(run_dir).resolve()
            allowed = (Path(config['_root'])/'runs/direction_a'/config['campaign']).resolve()
            if not run_dir.is_relative_to(allowed):
                raise ValueError('Training output escapes isolated campaign')
            invalid = run_dir/'input_invalid.json'
            if invalid.exists():
                raise InputDrift('Prior invalid input attempt requires a fresh run directory')
        try:
            with frozen_inputs(config):
                result = function(config, *args, **kwargs)
            return result
        except InputDrift as error:
            if invalid is not None:
                invalid.parent.mkdir(parents=True,exist_ok=True)
                invalid.write_text(json.dumps(dict(status='INVALID_INPUTS',result_claim_allowed=False,
                    error=str(error))),encoding='utf-8')
            raise
    return guarded


def assert_no_repair(path, *, expected=None):
    """Validate true JPEG/PNG contents, never infer encoding from the extension.

    Exact signatures and Pillow must agree. Verification plus a full decode are
    mandatory. Only actual JPEG contents undergo the JPEG-specific EOI check.
    """
    from PIL import Image
    path = Path(path)
    with path.open('rb') as stream:
        prefix = stream.read(16)
    if prefix.startswith(b'\xff\xd8\xff'):
        signature_format = 'JPEG'
    elif prefix.startswith(b'\x89PNG\r\n\x1a\n'):
        signature_format = 'PNG'
    else:
        raise ValueError(f'Unsupported or damaged image signature: {path}')
    with Image.open(path) as im:
        if im.format != signature_format:
            raise ValueError(f'Image signature/Pillow encoding mismatch: {path}')
        im.verify()
    with Image.open(path) as im:
        if im.format not in ('JPEG', 'PNG') or im.getexif().get(274) not in (None, 1):
            raise ValueError(f'Unreviewed image format/orientation: {path}')
        if min(im.size) <= 9:
            raise ValueError(f'Invalid image geometry: {path}')
        size = list(im.size)
        orientation = im.getexif().get(274)
        im.load()
    if expected is not None:
        if (signature_format != expected['format'] or size != expected['size_wh']
                or orientation != expected['exif_orientation']):
            raise ValueError(f'Image encoding/geometry differs from frozen manifest: {path}')
    if signature_format == 'JPEG':
        with path.open('rb') as stream:
            stream.seek(-2, 2)
            if stream.read() != b'\xff\xd9':
                raise ValueError(f'JPEG would trigger automatic repair; prohibited: {path}')
    extension = path.suffix.lower()
    extension_encoding = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG'}.get(extension)
    return dict(filename=path.name, extension=extension, signature_format=signature_format,
                pillow_format=signature_format, size_wh=size, exif_orientation=orientation,
                extension_content_mismatch=extension_encoding != signature_format,
                decode_status='PIL_VERIFY_AND_FULL_DECODE_PASS', sha256=sha(path))


def prepare_frozen_view(config, destination, train_rows, val_rows):
    """Class directories are hardlinks to the one canonical snapshot, never copies."""
    from .protocol import publish_contract
    root, manifest, data = binding(config)
    destination = Path(destination)
    if not destination.resolve().is_relative_to((root/'runs').resolve()):
        raise ValueError('Input view/output must stay under the isolated runs directory')
    mapping = {r['filename']:r for r in data['images']}
    image_root = (root / config['data']['images']).resolve()
    if image_root != (root / data['images_root']).resolve():
        raise ValueError('Configured image root differs from frozen snapshot')
    contract = dict(task='classifier',train=list(train_rows),val=list(val_rows),
        snapshot_manifest=str(manifest),snapshot_manifest_sha256=sha(manifest),
        release_root=str(root),dataset_id=data['dataset_id'],
        normalization_policy=data['normalization_policy'],
        canonical_images_root=str(image_root),link_type='HARDLINK_TO_CANONICAL_SNAPSHOT')
    publish_contract(destination / 'view_contract.json', contract)
    for role, rows in [('train',train_rows),('val',val_rows)]:
        for cls in (0,1):
            (destination / role / str(cls)).mkdir(parents=True, exist_ok=True)
        for row in rows:
            name = row['filename']; fixed = mapping[name]
            if any(row[k] != fixed[k] for k in ['label','fold','group_id']):
                raise ValueError(f'Image role/identity differs from canonical snapshot: {name}')
            source = image_root / name
            target = destination / role / str(row['label']) / name
            if target.exists():
                if not os.path.samefile(source, target):
                    raise ValueError('View is not a canonical snapshot hardlink')
            else:
                os.link(source, target)
            if sha(target) != fixed['sha256']:
                raise ValueError(f'View input byte drift: {target}')
    return destination

