"""
Dataset-validity scripts for CheoBench (thesis §4.1 + Appendix D).

Three sibling modules, each runnable standalone via ``python -m``:

- :mod:`dataset_quality` — coverage / integrity / discrimination / consistency
- :mod:`intrinsic` — TN-B1: Local/Community/Global has structural basis
- :mod:`convergent` — TN-B2: CheoBench scores correlate with user judgement

All three share paths and helpers from :mod:`._common`.
"""
