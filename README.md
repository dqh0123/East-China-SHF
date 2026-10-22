# East China SHF — thermal partitioning and model comparison

Code accompanying *Regional Thermal Partitioning of East China Constrained by
Multi-Source Data and Machine Learning*.

## Contents

| Path | Contents |
| --- | --- |
| `ASG-GKANWR/` | KAN vs. MLP model comparison pipeline (`models/`, `train_geothermal.ipynb`, `predict_geothermal.ipynb`, `file_io.py`) |
| `preAnalysis.ipynb` | exploratory analysis of the input data |
| `interpolation_pipeline.ipynb`, `interp_mincurv_east_china.ipynb` | interpolation and gridding of the station data |
| `unified_mesh.ipynb`, `quadtree_mesh.py` | adaptive quadtree meshing and distance-vector features |
| `analysisLine.ipynb`, `combined_3d_visual.ipynb` | profile and 3-D visualisation |
| `ASG-GKANWR/THIRD_PARTY_LICENSES/` | third-party notices and the full license text that applies here |

## Third-party code and citation

`ASG-GKANWR/models/` contains code from the BSRBF_KAN project:

> Ta, H. T. (2025). BSRBF-KAN: A combination of B-splines and Radial Basis
> Functions in Kolmogorov-Arnold Networks. In W. Buntine, M. Fjeld, T. Tran,
> M.-T. Tran, B. Huynh Thi Thanh, & T. Miyoshi (Eds.), *Information and
> Communication Technology* (pp. 3–15). Singapore: Springer Nature.
> <https://doi.org/10.1007/978-981-96-4288-5_1>

See
[`ASG-GKANWR/THIRD_PARTY_LICENSES/THIRD_PARTY_NOTICES.md`](ASG-GKANWR/THIRD_PARTY_LICENSES/THIRD_PARTY_NOTICES.md)
for the files involved and their licenses. Its license requires attribution as
"Based on software developed by Hoang-Thang Ta." and citation of the paper
above.
