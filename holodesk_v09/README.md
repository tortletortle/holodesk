# HoloDesk V0.9 — Motion-compensated Gray code

## Question
Can a temporal Gray-code sequence tolerate a moving hand/target if each frame is mapped back to a common reference coordinate using an independently estimated motion track?

## Method
- 256 discrete projector coordinates; eight Gray-code frames.
- Reference coordinate is sampled uniformly from [4, 251].
- Constant image-plane motion is tested at 0, 0.25, 0.5, 1, 2, and 4 pixels/frame.
- A noisy per-frame displacement track is fitted with a least-squares linear model.
- Each observation is mapped to the estimated reference coordinate before decoding.
- Baseline decodes the moving coordinate directly without compensation.
- Five seeds, 10,000 points per seed and condition.
- Metrics: MAE, median and 95th percentile error, within-1-pixel rate, catastrophic error rate (>8 px), and flow-fit RMSE.

Run `python benchmark.py`; NumPy is the only dependency.

## Interpretation and limitations
This is a synthetic 1-D coding experiment, not a real camera optical-flow implementation. The flow track is provided as noisy displacement observations; the study does not solve feature tracking, occlusion, non-rigid finger motion, perspective, projector-camera calibration, rolling shutter, defocus, or pattern decoding under real image formation. The model assumes constant velocity during the eight-frame sequence. The compensation method can be worse than the uncorrected baseline when the object is stationary but flow estimates are noisy. A production system therefore needs a motion-confidence gate and a non-compensated fallback.

Results are sensitivity evidence only; no real-world accuracy or latency claim is established.
