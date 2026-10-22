"""Native Ultralytics transforms/reads with immutable canonical image membership."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ultralytics.data.dataset import ClassificationDataset

from .immutable_inputs import assert_no_repair, verified_read, sha


class FrozenClassificationDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=''):
        if args.cache is not False or args.fraction != 1.0:
            raise ValueError('Frozen protocol requires cache=False, fraction=1.0')
        self.view = json.loads((Path(root).parent / 'view_contract.json').read_text(encoding='utf-8'))
        manifest = Path(self.view['snapshot_manifest'])
        if sha(manifest) != self.view['snapshot_manifest_sha256']:
            raise ValueError('View manifest drift')
        snapshot = json.loads(manifest.read_text(encoding='utf-8'))
        self.expected = {r['filename']:r for r in snapshot['images']}
        self.role_rows = self.view[Path(root).name]
        self.canonical_root = Path(self.view['canonical_images_root'])
        super().__init__(root=root, args=args, augment=augment, prefix=prefix)

    def verify_images(self):
        """Never call a mutating repair or silently filter an identity."""
        wanted = {r['filename']:int(r['label']) for r in self.role_rows}
        actual = {}
        if self.base.class_to_idx != {'0':0,'1':1}:
            raise ValueError('Frozen class semantics differ')
        for filename, label in self.samples:
            path = Path(filename); name = path.name
            if name in actual or name not in wanted or label != wanted[name]:
                raise ValueError('Classification image membership/label differs')
            canonical = self.canonical_root / name
            if not os.path.samefile(path, canonical):
                raise ValueError('Classification must read the same canonical file')
            with verified_read(path, self.expected[name]['sha256']):
                assert_no_repair(path, expected=self.expected[name]['decode_validation'])
            actual[name] = label
        if actual != wanted:
            raise ValueError('Classification source coverage is incomplete')
        return self.samples

    def __getitem__(self, index):
        path = Path(self.samples[index][0])
        # Works also in Windows spawn workers; no open OS handle is pickled.
        with verified_read(path, self.expected[path.name]['sha256']):
            return super().__getitem__(index)


def build_classification_dataset(trainer, img_path, mode='train', batch=None):
    # Constructor and augmentation are inherited byte-for-byte from pinned Ultralytics.
    return FrozenClassificationDataset(root=img_path,args=trainer.args,
                                       augment=mode=='train',prefix=mode)

