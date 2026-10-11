# Publishing this repository to GitHub

This folder is already an initialised Git repository with one commit on the `main`
branch, so publishing is just *create remote → push*.

## 1. Create the empty remote

On GitHub: **New repository → name it `building-retrofit-moo-climate` → Public →
do NOT add a README/.gitignore/license** (they already exist here) → Create.

## 2. Point the local repo at it and push

Replace `your-username` with your GitHub username:

```bash
git remote add origin https://github.com/your-username/building-retrofit-moo-climate.git
git push -u origin main
```

## 3. Optional: install the GitHub CLI for a one-liner

```bash
gh auth login
gh repo create building-retrofit-moo-climate --public --source=. --remote=origin --push
```

## After pushing

- Update the placeholder `your-username` in `README.md`, `pyproject.toml` and
  `CITATION.cff` to your real GitHub URL.
- Add repository **topics** on GitHub (e.g. `multi-objective-optimization`,
  `building-energy`, `nsga2`, `topsis`, `shap`, `climate-change`) for discoverability.
- The `results/figures/*.png` files render directly in the README.

## Notes

- The commit includes ~26 MB of figures/tables. That is well under GitHub's limits.
  If you prefer a lean repository, remove `results/` from version control and let
  users regenerate it with `python scripts/run_pipeline.py`.
- `git` needs `core.longpaths true` on Windows because the project folder name can be
  long; this is already set locally for this repository.
