# Thesis LaTeX Workspace

This folder is a local-first thesis workspace.
Current template is prepared for Slovak writing and aligned with
EU Bratislava internal directive IS 1/2024 (recommended formatting rules).

## Prerequisites

1. TeX distribution:
- TeX Live (recommended), or
- MiKTeX
2. `latexmk` and `biber` available in `PATH`
3. Perl runtime (required by `latexmk`; Strawberry Perl on Windows)

For Windows with MiKTeX + Strawberry Perl, `build.ps1` also tries common
install paths automatically, so build can work even before shell PATH refresh.

## Quick Start (Windows PowerShell)

```powershell
cd thesis
.\build.ps1 build
```

Compiled PDF will be written to:

```text
thesis/build/main.pdf
```

Auto-rebuild while editing:

```powershell
.\build.ps1 watch
```

Clean artifacts:

```powershell
.\build.ps1 clean
.\build.ps1 distclean
```

## Files You Will Edit First

1. `thesis/metadata.tex` (title, author, supervisor)
2. `thesis/chapters/*.tex` (thesis text)
3. `thesis/references.bib` (literature)

## Current Content Status

- Template includes required front matter:
  - cover (`obal`)
  - title page (`titulný list`)
  - Slovak abstract
  - English abstract
  - table of contents
- Main body is intentionally a blank chapter skeleton for writing.
- Registered assignment details are prefilled in `thesis/metadata.tex`.

## Notes

1. Update placeholders in `thesis/metadata.tex`:
   - `\EvidenceNumber`
   - `\StudyFieldName`
2. If your faculty gives stricter rules beyond IS 1/2024, adjust
   `thesis/main.tex` and front matter files accordingly.
