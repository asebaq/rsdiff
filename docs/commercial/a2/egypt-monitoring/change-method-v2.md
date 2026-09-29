# Construction-aware change baseline v2

Frozen 2026-09-03 before running on the held-out `new_cairo_mostakbal` AOI. New
Capital v1 review decisions motivated the design but are not reused as test labels.

## Fixed method

1. Estimate after-to-before registration on stable, high-gradient B04 pixels using an
   exhaustive ±1.5 pixel search in 0.25-pixel steps. Minimize mean absolute residual
   after removing the candidate shift's median radiometric offset. Reject a solution
   on the search boundary or with displacement above 1.5 pixels.
2. Shift after-date reflectance bilinearly and SCL with nearest-neighbour resampling.
   Exclude invalid SCL classes and shift-created borders.
3. Compute the v1 four-band robust RMS score after registration.
4. Suppress pixels where either date has NDVI above 0.25. Require absolute visible
   brightness change of at least 0.03 reflectance and robust score at least 4.5.
5. Aggregate fragments with a 30 m binary closing. Record original thresholded area
   separately from the resulting grouping envelope; filled pixels are not observations.
6. Apply mutually exclusive class gates:
   - road formation: at least 6,000 m², bbox elongation at least 3, median before
     brightness at least 0.22, and median brightness change at most -0.02;
   - structural footprint: at least 10,000 m² and bbox fill at least 0.15;
   - clearing/grading: at least 50,000 m²;
   - laydown/stockpile: at least 20,000 m²;
   - otherwise discard.
7. Export every retained aggregate as `needs_review`. Mechanical confidence remains a
   score description, not construction confidence.

## Evaluation

The held-out usefulness gate remains 80%. Every proposal is reviewed against observed
before/after imagery. No parameter may change after viewing Mostakbal proposals. A
future independent test is still required because the same reviewer designed and
evaluates this experiment.
