# Project site (MkDocs Material)

Source for [asebaq.github.io/rsdiff](https://asebaq.github.io/rsdiff).

## Local preview

```bash
uv pip install -e ".[docs]"
mkdocs serve -f docs/site/mkdocs.yml
# open http://127.0.0.1:8000
```

## Build static site

```bash
mkdocs build -f docs/site/mkdocs.yml --strict --site-dir site_out
```

## Deploy

GitHub Actions (`.github/workflows/docs.yml`) builds and pushes to the `gh-pages` branch on every push to `main`. Custom domain set via `extra.url` in `mkdocs.yml` plus a `CNAME` file in `docs/`.

## Layout

```
docs/site/
  mkdocs.yml             # config
  docs/
    index.md             # landing
    method.md            # architecture + recipe
    results.md           # FID table + sample grids (placeholders until v0 lands)
    usage.md             # install + sampling + training
    citation.md          # bibtex
    assets/              # images, sample grids (gitignored heavies)
```

## Figures

Figures live under `docs/site/docs/figures/` (copied from the repo-level `docs/figures/` so MkDocs can serve them). Regenerate with `python scripts/make_plots.py`.
