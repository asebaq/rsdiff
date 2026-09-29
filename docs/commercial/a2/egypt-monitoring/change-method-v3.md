# Three-date persistence baseline v3

Frozen 2026-09-03 before inspecting proposals in `new_capital_east_holdout`.

## Fixed method

1. Generate 2024→2025 proposals with unchanged `construction_aware_v2` parameters.
2. Register 2025 to 2024 and 2026 to 2025 with the v2 registration search. Compose
   both shifts so all reflectance is on the 2024 reference grid.
3. Within each v2 aggregate, reconstruct its original evidence pixels using the frozen
   v2 score, brightness, NDVI, and validity gates.
4. For each evidence pixel, calculate four-band Euclidean spectral distance from 2026
   to 2024 and from 2026 to the registered 2025 state.
5. A pixel persists when its 2026 state remains at least 70% as far from 2024 as the
   2025 state did, and 2026 is closer to 2025 than to 2024.
6. Retain a proposal when at least 65% of valid evidence pixels persist. Export the v2
   class unchanged and add persistence counts/fraction. All proposals remain
   `needs_review`.

This method only validates persistence of 2024→2025 proposals. It does not discover
new work starting in 2025→2026. The 80% usefulness gate remains unchanged and no
parameter may change after reviewing the holdout.
