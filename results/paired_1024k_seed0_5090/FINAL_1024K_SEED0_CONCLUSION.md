# 1024 kimg seed0 paired evaluation conclusion

Status: **MIXED_OR_TIE**

## Optimization-loss evidence

Adaptive-B final loss was 17.38587940 versus 16.77903271 for fixed (+3.617% relative). Fixed completed 7987 successful updates with 13 AMP skips; Adaptive-B completed 7985 with 15 skips. Neither trajectory contains a non-finite recorded loss. Training loss is not treated as generation-quality evidence.

## Generation-quality evidence

KID deltas (adaptive minus fixed) are +0.000449683 at NFE1 and -0.000327659 at NFE2; FID deltas are +1.136762 and +0.194039. The paired metrics are mixed or effectively tied under the pre-frozen directional rule.

## Visual evidence

Unselected paired seeds 0–63 are archived for both schedules and both NFEs. They are auxiliary and do not override KID/FID.

## Decision and limitations

This is a strictly paired RTX 5090, 1024 kimg, training-seed0 result. It cannot establish general superiority or inferiority. The metrics use 5k generated samples, not standard FID-50k. Seed1 uses a separate batch-gpu=128 execution protocol and must be compared within its own paired arms.
