# Construction-aware v2 result

Run and reviewed 2026-09-03 on held-out `new_cairo_mostakbal` using the parameters
frozen in [`change-method-v2.md`](change-method-v2.md). Parameters were not changed
after viewing the held-out candidates.

## Result

- estimated after-to-before shift: -0.25 row, +0.25 column (about 2.5 m per axis);
- valid 10 m pixels: 448,077;
- retained proposals: 17, compared with 27 from v1 on New Capital;
- accepted: 13;
- rejected: 4;
- usefulness: **76.47%**, compared with 18.52% for v1;
- accepted types: 8 structural footprints, 4 road formations, and 1 laydown or
  stockpile change.

The improvement is large but the method still fails the frozen 80% gate. Remaining
false positives are broad disturbed sand and ambiguous bare-surface changes. This run
must not be reclassified or retuned to cross the threshold.

## Interpretation

Registration, vegetation suppression, aggregation, and class-specific gates addressed
the dominant v1 failure modes. However, this is still a single-AOI evaluation by the
same reviewer who designed the method. It is evidence that the direction is promising,
not a product-quality accuracy claim.

The next defensible iteration is persistence across a third date: require a structural
or road signal to remain or develop coherently rather than accepting a two-date bare-
surface transition. Freeze that method and evaluate it on a new AOI. Do not develop a
plugin yet.
