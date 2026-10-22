# Focal Sampling adaptation

The paper uses a fixed-weight adaptation of the Focal Sampling method of Quercia and collaborators. The relevant reviewed upstream source is `https://github.com/AlessioQuercia/sgd_biased` at commit `438dbe399ca1f7f623879ece45f361d3ba906e8c`, with Apache-2.0 licensing and the copyright/author credits stated in `THIRD_PARTY_NOTICES.md`.

The local implementation is in `focal_sampling.py`, `focal_training.py`, and `focal_loader.py`. It records the last training-batch true-class probability for each FIT image, freezes weights proportional to `(1-p_true)^0.5` after the 11th completed epoch, and samples FIT identities with replacement in later epochs. This is a sampling adaptation with cross-entropy optimization, not Focal Loss. The R6 execution snapshot also includes input-snapshot guards and a frozen classification-dataset adapter; see the source and configuration.

Attribution comments retained from the earlier public package do not alter the executable syntax tree. A prior bounded line comparison of the old package with 23 upstream Python files is historical evidence only; it is not a new legal determination for every R6 source file. File-level source and licence review remains required before publication.
