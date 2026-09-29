# Example construction-change deliverable

This is the output contract for a manual concierge pilot. It is intentionally independent
of a QGIS plugin and RSDiff.

## Required inputs

- Earlier and later orthorectified GeoTIFFs covering the same AOI.
- CRS, acquisition dates, provider, ground sample distance, and usage rights.
- AOI polygon and optional cadastral/building baseline.
- Customer definition of relevant change and minimum mapping unit.
- Required language, output CRS, and confidentiality/retention rules.

If imagery is not sufficiently aligned, registration quality is reported before change
analysis. The system must not silently interpret misregistration as construction.

## Package

```text
project-id/
├── README.md
├── provenance.json
├── changes.gpkg
├── changes.geojson
├── review.csv
├── evidence/
│   ├── change-0001-before.png
│   ├── change-0001-after.png
│   └── ...
└── report/
    ├── report-en.pdf
    └── report-ar.pdf
```

## Change-layer schema

| Field | Meaning |
|---|---|
| `change_id` | Stable project-local identifier |
| `change_type` | `new`, `expanded`, `removed`, or customer-defined class |
| `area_m2` | Area in an appropriate projected CRS |
| `confidence` | Calibrated model score, not a probability unless validated |
| `review_status` | `pending`, `accepted`, `rejected`, or `edited` |
| `reviewer` | Customer-provided reviewer identifier, optional |
| `before_date` / `after_date` | Imagery acquisition dates |
| `source_ids` | References into provenance metadata |
| `notes` | Analyst correction or ambiguity |

Every geometry must remain editable. Automated output is a proposal until reviewed.

## English report outline

1. Executive summary: AOI, period, reviewed change counts, and affected area.
2. Inputs and rights: imagery sources, dates, resolution, CRS, and declared permissions.
3. Method: registration, inference, vectorization, thresholds, and human review.
4. Quality: alignment error, test sample, false-positive/negative estimates, limitations.
5. Results: map, table by change type, and evidence panels.
6. Review log: accepted, rejected, and edited proposals.
7. Provenance: software/model versions and processing timestamps.

## Arabic report headings

- الملخص التنفيذي
- مصادر البيانات وحقوق الاستخدام
- منهجية المعالجة والتحقق
- تقييم الجودة والقيود
- نتائج رصد التغيرات
- سجل المراجعة البشرية
- بيانات المصدر وإصدارات النماذج

Arabic content must be reviewed by a fluent domain professional before customer use;
machine translation alone is not an acceptance criterion.

## Pilot measurements

- baseline manual minutes versus assisted minutes;
- registration residual in pixels/metres;
- proposal precision and recall on a customer-reviewed sample;
- accepted, rejected, and edited proposal counts;
- median correction time per proposal;
- report preparation time;
- imagery/software cost and compute time;
- customer acceptance or reasons for rejection.

## Acceptance gate

The example passes only if outputs open correctly in QGIS, retain the requested CRS,
preserve provenance, support edit/review, and reduce total analyst time by at least 50%
without hiding false negatives. Exact accuracy thresholds must be agreed per pilot before
processing.

