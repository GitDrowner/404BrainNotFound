# Per-transform operating thresholds: aligned 640-image run

This file records the provenance and limitations of the operating thresholds loaded by the local
773086 demo backend. It contains no dataset images or per-image scores.

## Protocol

- Model: frozen job-773086 checkpoint and frozen FP32 Platt temperature/bias.
- Source: the internal test corpus, sampled deterministically as 320 COCO real images and 320
  MidJourney fake images. The previous 64-image subset is fully contained in this set.
- Input alignment before each selected competition transform: RGB, direct Bicubic resize to
  256×256, then JPEG quality 95 with 4:2:0 chroma subsampling.
- Evaluation: 640 images × 16 transforms = 10,240 recorded forwards.
- Objective: choose a separate score boundary per transform that maximizes accuracy on these test
  labels.

The source manifest SHA-256 is
`9b6f8c4ca0de3b96d287ffda7ba79dca35e6a858af86b47021f52e8e0151fac9`. The local per-image
audit SHA-256 is `4add8ba662fac7fa97d4057b141036e3223fb09f9e42bc448fe53eae4787bec6`.
Per-image data is intentionally excluded from Git.

## Result

Across the 16 conditions, mean AUROC was 94.67%. Mean accuracy at the old common threshold
`0.2815194250` was 85.10%; post-hoc per-transform thresholds increased it to 87.91%. The weakest
oracle accuracy was 84.06% under Gaussian noise σ=0.10.

| Transform | Threshold | AUROC | Oracle accuracy |
|---|---:|---:|---:|
| clean | 0.428257 | 0.9529 | 89.22% |
| JPEG Q90 | 0.497436 | 0.9426 | 87.50% |
| JPEG Q70 | 0.600460 | 0.9574 | 88.75% |
| JPEG Q50 | 0.398548 | 0.9487 | 87.66% |
| JPEG Q30 | 0.403625 | 0.9506 | 88.44% |
| blur σ=0.5 | 0.378077 | 0.9563 | 90.00% |
| blur σ=1.0 | 0.518252 | 0.9489 | 87.97% |
| blur σ=2.0 | 0.383400 | 0.9477 | 87.97% |
| resize 0.5× | 0.388286 | 0.9597 | 89.06% |
| resize 0.25× | 0.416726 | 0.9332 | 85.62% |
| noise σ=0.02 | 0.276947 | 0.9550 | 88.91% |
| noise σ=0.05 | 0.417445 | 0.9398 | 86.88% |
| noise σ=0.10 | 0.394505 | 0.9100 | 84.06% |
| color −20% | 0.561951 | 0.9459 | 87.97% |
| color +20% | 0.343851 | 0.9556 | 88.59% |
| center crop 80% | 0.492106 | 0.9436 | 87.97% |

## Interpretation boundary

These values are post-hoc test-label oracle operating points, not an unbiased held-out test result
and not a replacement for the Platt probability calibration. They are enabled for the local demo
at the user's request. `probability_fake` retains the frozen Platt score; `aigc_confidence` only
maps the selected transform's configured boundary to the display value 0.5. Performance on unseen
commercial generators can remain substantially worse because transform robustness does not imply
cross-generator robustness.
