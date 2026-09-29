# License audit

Status: preliminary, 2026-08-28. This is an engineering inventory, not legal advice.

The repository license does not automatically grant rights to datasets, pretrained
models, generated datasets, third-party dependencies, or imagery supplied by customers.

| Component | Current evidence | Commercial status | Required action |
|---|---|---|---|
| RSDiff-authored repository code | Root `LICENSE`: Apache-2.0 | Permissive for the authored code | Preserve notices and identify any copied third-party files |
| Released `rsdiff1.5` weights | HF model card declares Apache-2.0 | **Not cleared** because training-data rights are unresolved | Confirm the author can license weights this way after resolving RSICD terms |
| `imagen-pytorch` | External dependency; project commonly distributed under MIT | Likely permissive, not covered by this repo's license | Pin the exact source/version and retain its license notice |
| T5-base weights/tokenizer | External Hugging Face/Google artifact | Expected Apache-2.0; exact artifact must be checked | Record model revision and bundled license before a commercial release |
| RSICD images and captions | Research dataset/HF mirror; no grant is recorded in this repository | **Blocking unknown** for commercial training, redistribution, and derived weights | Locate original dataset terms and obtain written clarification if necessary |
| Evaluation models (`clean-fid`, OpenCLIP) | External packages and model weights | Separate licenses and use terms apply | Record exact packages, weight sources, and notices |
| Customer satellite/aerial imagery | Provider- and contract-specific | **Unknown per customer** | Require written rights for processing, retention, derived layers, and model improvement |
| Sentinel-2 L2A (Earth Search COGs) | Copernicus Sentinel data legal notice | Free use, including commercial, with attribution | Credit "contains modified Copernicus Sentinel data [year]" in every deliverable |
| Sentinel-1 RTC (Microsoft Planetary Computer, Catalyst processing) | Collection metadata declares CC BY 4.0; derived from Copernicus Sentinel-1 GRD | Commercial use permitted with attribution | Credit Copernicus and Catalyst RTC processing; recheck terms if moving beyond anonymous SAS-token reads |
| QGIS plugin | QGIS/Python plugin ecosystem | GPL compatibility and distribution obligations require review | Decide open plugin vs separate proprietary service only after legal review |
| Stable Diffusion or other future base models | Not part of the released cascade | Model-specific license applies | Audit before adoption; never infer rights from `diffusers`' Apache license |

## Phase B1 decision

The code can remain Apache-2.0. Do not market the checkpoint, RSICD-derived synthetic
data, or a future commercial service as commercially cleared until RSICD's original
license and image provenance are documented. Do not use customer imagery for training
or evaluation by default.

## Before any paid pilot

- Define who owns input imagery, derived polygons, reports, and generated imagery.
- Define retention, deletion, hosting region, and whether inference is local or cloud.
- Disable training on customer data unless separately opted in with explicit terms.
- Record every model and dataset revision in the delivered system.
- Add synthetic-content provenance when generated imagery is exported.

