# HoloDesk V0.8 — Temporal structured-light code study

## Research question
Compare temporal Gray-code, randomized 8-bit codewords, and 3-step phase-shift decoding under pixel noise and per-frame motion.

## Model
- One-dimensional projector coordinate with 256 addressable positions.
- Gray code: 8 binary frames.
- Random code: 8 binary frames, a shuffled permutation of all 8-bit words.
- Phase shift: two periods (32 and 256 pixels), three phase steps each; the 256-pixel period spans the coordinate range.
- Motion: the observed point shifts linearly between projected frames.
- Observation: ambient offset + pattern contrast * ideal pattern + Gaussian noise. The noise standard deviation includes a simple ambient-dependent term: sqrt(read_noise^2 + 0.01 * ambient).
- Black/white calibration and ambient subtraction are assumed known exactly.

Run `python benchmark.py`; NumPy is the only dependency. The script writes `results.csv`.

## Interpretation limits
This is a 1-D synthetic decoder comparison, not a camera-projector stereo reconstruction and not a physical optical simulation. It omits projector gamma, defocus, surface reflectance, shadows, rolling shutter, auto-exposure, saturation, calibration error, occlusion, and real motion blur. Ambient calibration is idealized. The randomized codebook has no designed error-correcting distance; it is included as a naive baseline. The phase decoder uses only the full-range phase in the final coordinate estimate; the coarse phase is retained as a possible future unwrap/consistency cue. Do not interpret these results as real-world touch accuracy or latency.
