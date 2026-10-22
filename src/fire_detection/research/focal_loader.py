# Public candidate adaptation notice (2026-10-18): this local finite loader
# supports the study's Ultralytics Focal Sampling adaptation. The published
# method and upstream implementation are by Alessio Quercia and collaborators.
# Upstream: https://github.com/AlessioQuercia/sgd_biased at
# 438dbe399ca1f7f623879ece45f361d3ba906e8c (Apache-2.0).
# Relevant upstream callbacks.py credits Copyright 2024 Forschungszentrum
# Juelich GmbH and file author Alessio Quercia. See FOCAL_ADAPTATION_NOTICE.md.
"""Finite train loader with the Ultralytics reset interface."""

from torch.utils.data import DataLoader


class FocalDataLoader(DataLoader):
    def reset(self):
        # Classification has no mosaic. Every epoch already creates a new finite
        # iterator and new workers; there is no cached InfiniteDataLoader iterator.
        if self.persistent_workers:
            raise RuntimeError("Focal loader must not retain worker iterators across epochs")
