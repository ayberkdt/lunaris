# Ephemeris interpolation validation

Run the DE440/SPICE off-node and DOP853 comparison with:

```bash
python validation/ephemeris/interpolation_validation.py \
  --kernel-dir data/ephemeris_models \
  --grid-step-s 60 \
  --output validation/ephemeris/interpolation_validation_2026_07_18.json
```

The JSON records kernel SHA-256 values, cadence, seed, direct-SPICE position
errors, DOP853 function evaluations, and final-position difference. Its scope is
limited to that configuration; it is validation evidence for the interpolation
change, not a general orbit-accuracy claim.

