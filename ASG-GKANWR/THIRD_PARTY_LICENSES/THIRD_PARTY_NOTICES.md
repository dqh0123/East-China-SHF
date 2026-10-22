# Third-party notices

This repository contains code taken from the **BSRBF_KAN** project. This file
sits next to the third-party license text, in `ASG-GKANWR/THIRD_PARTY_LICENSES/`.

## BSRBF_KAN

| Item | Value |
| --- | --- |
| Upstream | <https://github.com/hoangthangta/BSRBF_KAN> |
| Author | Hoang-Thang Ta |
| Associated paper | Ta, H. T. (2025). *BSRBF-KAN: A combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks*. In W. Buntine, M. Fjeld, T. Tran, M.-T. Tran, B. Huynh Thi Thanh, & T. Miyoshi (Eds.), *Information and Communication Technology* (pp. 3–15). Singapore: Springer Nature. <https://doi.org/10.1007/978-981-96-4288-5_1> (preprint: arXiv:2406.11173) |
| Attribution required | "Based on software developed by Hoang-Thang Ta." |

### License

The copy vendored here was taken from an upstream snapshot distributed under the
**MIT License** (Copyright (c) 2024 Hoang Thang Ta).

The project's *current* license is the **Academic and Non-Commercial License**
(Copyright (c) 2026 Hoang-Thang Ta). This repository complies with its terms as
well: use is limited to educational, academic-research and other non-commercial
research purposes, attribution is retained, and any academic work that uses this
code must cite the associated paper. The complete license text is kept next to
this file in [`BSRBF_KAN-LICENSE.txt`](BSRBF_KAN-LICENSE.txt) and must accompany
any redistribution.

### Files

Files in `../models/` (i.e. `ASG-GKANWR/models/`) that come from BSRBF_KAN:

| File | Status in this repository | Notes |
| --- | --- | --- |
| `bsrbf_kan.py` | unmodified | attribution header added at the top |
| `mlp.py` | unmodified | attribution header added at the top |
| `efficient_kan.py` | unmodified (provenance header added) | original notice retained ("modified from <https://github.com/Blealtan/efficient-kan>", MIT) |
| `fast_kan.py` | unmodified (provenance header added) | original notice retained: Copyright 2024 Li, Ziyao, **Apache License 2.0** |
| `faster_kan.py` | unmodified (provenance header added) | no separate notice in the upstream distribution |
| `gottlieb_kan.py` | unmodified (provenance header added) | original notice retained ("modified from <https://github.com/seydi1370/Basis_Functions>") |
| `__init__.py` | **modified** | added `from .rbf_kan import RBFKAN` and the matching `__all__` entry for the RBFKAN variant implemented in this repository |

`rbf_kan.py` is original code of this repository (a simplified SiLU+RBF variant,
inspired by the BSRBF-KAN layer design) and is not a copy of upstream code.

### Modification statement

Aside from the `__init__.py` change listed above and comment-only provenance
headers added at the top of the six copied model files, no functional changes
were made to the third-party files. All of them still carry their own upstream
notices, and `fast_kan.py` remains under the Apache License 2.0, which this
notice does not narrow.
