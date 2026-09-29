# Online candidate review

Reviewed 2026-09-02. This is desk research, not customer validation. No organization
was contacted and no source below is treated as an endorsement.

## Decision

Start with **independent construction-milestone verification from repeat observed
imagery**. Do not build a general change-detection product or QGIS plugin yet.

Keep **synthetic rare-class augmentation** as the first RSDiff research experiment,
not as the first commercial product. Defer **unauthorized-building detection** until
there is legal access to a municipality's permit/building register and suitable
high-resolution imagery.

## Scoring method

Each criterion is scored 1 (poor) to 5 (strong). Equal weights are deliberate: current
evidence is not good enough to justify precise market-size weights. Scores express the
decision from a solo-founder, one-week-validation perspective; they are not TAM
estimates.

| Criterion | Construction milestone verification | Unauthorized buildings | Dataset QA / synthetic augmentation |
|---|---:|---:|---:|
| Observable pain | 5 | 5 | 3 |
| Identifiable buyer | 4 | 5 | 3 |
| Recurring frequency | 5 | 4 | 3 |
| Evidence of existing spend | 4 | 4 | 2 |
| Competition opportunity | 3 | 3 | 3 |
| Affordable public demo data | 4 | 2 | 5 |
| Licensing path | 4 | 2 | 2 |
| Founder/repository fit | 5 | 4 | 5 |
| One-week validation feasibility | 5 | 2 | 4 |
| **Total / 45** | **39** | **31** | **30** |

“Competition opportunity” rewards a credible opening, not absence of competitors.
Existing vendors validate demand but make a generic feature set unattractive.

## Evidence by candidate

### 1. Construction milestone verification — winner

The recurring job is visible: a funder, owner, insurer, or portfolio manager needs an
independent record of whether a large site reached an observable milestone without
depending only on a contractor update or site visit.

- [GammaEarth](https://www.gamma.earth/industries/construction/) markets repeat site
  monitoring, historical comparisons, portfolio workflows, GeoTIFF output, and a
  public price anchor of €2/km² with a €200 starting balance. This validates both the
  workflow and low-cost imagery demand. Its claimed 1 m output is model-derived
  super-resolution of 10 m Sentinel-2 data, so it must not be presented as new observed
  spatial detail.
- [FAS's Ajman and Memphis case studies](https://fas.org/publication/tracking-hyperscale/)
  show a concrete independent-verification job: public announcements and schedules
  were compared with activity visible in satellite imagery. The report also emphasizes
  that imagery is one source in a wider verification toolkit, not definitive proof of
  every construction state.
- [Off-Nadir Delta](https://offnadir-delta.com/blog/construction-monitoring-satellite-imagery)
  and [SATPALDA](https://satpalda.com/construction-change-detection-using-high-resolution-satellite-imagery/)
  offer similar monitoring, confirming competition and arguing against a generic
  “change detection” product.
- The [Sat4BIM4D study](https://rgg.edu.pl/pdf-196715-119525?filename=119525.pdf)
  explores integration of BIM and satellite remote sensing, indicating that detailed
  progress percentages require project records and stronger imagery—not pixels alone.

The narrow opening is evidence packaging: registered observed dates, explicit visible
milestones, editable review polygons, provenance, uncertainty, and an Arabic/English
brief. Initial targets should be large MENA industrial, infrastructure, earthworks, or
data-center sites where 10 m public imagery can reveal macro changes. Small buildings,
interior work, and contractual completion percentages are out of scope.

### 2. Unauthorized-building detection — real need, poor first entry

- ESA's [DICAS activity](https://incubed.esa.int/portfolio/illegal-construction-detector/)
  explicitly targets municipalities, undeclared construction, lost property-tax
  revenue, and labor-intensive complaint/manual-inspection workflows. It compares
  detections from 30 cm satellite imagery with building-register data and names a web
  application for follow-up actions.
- The same source calls the activity ongoing/de-risking and says product development
  will most likely continue. This is evidence of a funded problem and a competitor,
  not proof of repeat purchases or product-market fit.
- Public studies demonstrate technical feasibility, including
  [new-construction identification](https://www.mdpi.com/2072-4292/14/13/3227) and
  [multi-temporal illegal-building detection](https://isprs-archives.copernicus.org/articles/XL-1-W5/387/2015/).

The decisive constraints are external: legality is not inferable from imagery alone;
it requires current permit/cadastral records. The ESA implementation also requires
30 cm commercial data. Municipal procurement, data-sharing, notice/appeal processes,
and false-positive consequences make an online-only one-week pilot misleading.

### 3. Dataset QA / synthetic augmentation — research track

- A 2026 [DDPM augmentation preprint](https://arxiv.org/abs/2608.16380) evaluates only
  on real held-out imagery and reports balanced accuracy improving from 67% to 81% and
  macro-F1 from 65% to 78% for battle-damaged agricultural fields. This is the right
  experimental pattern, but one preprint is not buying evidence.
- A [rare-object generation study](https://arxiv.org/html/2409.01138v1) finds that
  synthetic rare-object satellite imagery is feasible, while common automated image
  metrics can correlate poorly with human judgement. This argues for downstream and
  human evaluation rather than FID-only claims.
- [Geo-typical synthetic building data](https://arxiv.org/abs/2507.16657) reports
  median improvements up to 12% using target-layout knowledge, procedural rendering,
  and domain adaptation. It also shows that useful synthetic data needs labels and
  conditioning, not only text-to-image samples.

This candidate fits RSDiff technically but lacks a clearly evidenced buyer and repeat
budget. Commercial use is additionally blocked until training-dataset and generated-
output rights are resolved. Run it as Track B3 only after baseline reproduction.

## One-week online validation demo

The demo tests whether a useful evidence packet can be produced—not whether a complete
product can be built.

### Fixed question

“Can public repeat imagery support a reviewable, provenance-preserving report of macro
construction milestones for one large MENA site?”

### Scope and acceptance

- One large site with visible earthworks or structural footprint changes and at least
  two low-cloud Sentinel-2 L2A dates from Copernicus Data Space.
- Observed 10 m bands remain the source of truth. No generated or super-resolved pixels
  are used as evidence.
- Register the dates; show before/after and a transparent change layer.
- Propose polygons for only predeclared classes: clearing/earthworks, large new roof or
  slab footprint, and large laydown-area change.
- Every polygon records source, acquisition date, method, confidence, and review state;
  a human can edit or reject it in QGIS.
- Export GeoJSON or GeoPackage plus a short English/Arabic PDF or Markdown brief.
- Success: a reviewer can trace every claim to imagery, edit the geometry, and classify
  at least 80% of proposed polygons as useful. Report false positives and omissions;
  do not tune the threshold after seeing the score.

### Daily sequence

1. Choose the AOI and write the observable-milestone rubric before viewing all dates.
2. Download and record imagery identifiers, dates, cloud cover, license/terms, CRS, and
   hashes; create registered true-color and index layers.
3. Implement the smallest deterministic change baseline and polygon extraction.
4. Review in QGIS; add confidence, rejection reasons, and provenance fields.
5. Produce the bilingual evidence brief with before/after crops and limitations.
6. Blind-review the proposals against the fixed rubric and calculate usefulness,
   false-positive count, omissions, processing time, and manual correction time.
7. Write the result and choose: repeat with a second site, change the target resolution,
   or kill the hypothesis. Do not start plugin development from one demo.

## What online research can and cannot decide

Online research is sufficient to reject crowded generic ideas, identify buyers,
document competitors and price anchors, and select this demo. It cannot establish the
buyer's acceptance rules, willingness to pay, internal data access, procurement path,
or whether the report replaces paid work. Interviews are therefore postponed, not
treated as a prerequisite for this demo. They become necessary only before calling the
idea commercially validated or investing in a product/plugin.
