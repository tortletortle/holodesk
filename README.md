# HoloDesk

Research prototype for low-cost RGB-camera tabletop interaction.

## Current repository status
- `holodesk_v01/benchmark.py`: synthetic optical geometry benchmark.
- `holodesk_v02/benchmark.py`: uncertainty-aware stationary-contact benchmark.
- `holodesk_v03/benchmark.py`: dynamic contact tracking benchmark.
- `holodesk_v04/geometry.py` and `holodesk_v05/geometry.py`: shared synthetic geometry utilities.
- `holodesk_v06/benchmark.py`: synthetic side-view contour-feature sensitivity experiment.

Historical source archives V0.1–V0.6 are retained separately in the project working artifacts. V0.4/V0.5 full benchmark scripts and all historical CSV files still need to be migrated into this repository. Do not treat the repository as a complete archive until that migration is finished.

## Reproducibility and evidence limits
All current experiments are synthetic. They do not establish performance on a physical camera/projector system. The V0.5 auxiliary cue is generated from simulation labels and is a sensitivity study, not a visual feature extractor. V0.6 uses an idealized clipped-ellipse image generator, not real finger images. No real-world touch accuracy, latency, or robustness claim is established.

## Environment
Python 3.11 recommended. Individual scripts require NumPy; SciPy for V0.2–V0.5; OpenCV, pandas and scikit-learn for V0.6. Run each benchmark from its own version directory so relative imports and output CSV paths resolve correctly.
