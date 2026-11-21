# HoloDesk V0.7 — Active optical correspondence

This is a synthetic pinhole camera/projector triangulation sensitivity study. It evaluates how pixel correspondence noise changes with ambient illumination and projected-pattern contrast.

It does NOT simulate a physical projector's spectral response, rolling shutter, surface BRDF, shadow formation, projector gamma, camera auto-exposure, pattern decoding failures, calibration drift, occlusion, or hand deformation. The noise law is an explicit simplifying assumption:
`sigma_px = 0.25 * sqrt(1 + ambient_ratio) / pattern_contrast`.

Each grid cell averages five random seeds, 10,000 3D points per seed. Metrics are 3D and height reconstruction errors in millimetres. This is a sensitivity study, not evidence of real-world performance and not a comparison against a real passive-shadow implementation.

Run: `python active_optics_benchmark.py`
Dependency: NumPy.
