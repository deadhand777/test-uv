# test-uv

[![Release](https://img.shields.io/github/v/release/deadhand777/test-uv)](https://img.shields.io/github/v/release/deadhand777/test-uv)
[![Build status](https://img.shields.io/github/actions/workflow/status/deadhand777/test-uv/main.yml?branch=main)](https://github.com/deadhand777/test-uv/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/deadhand777/test-uv)](https://img.shields.io/github/commit-activity/m/deadhand777/test-uv)
[![License](https://img.shields.io/github/license/deadhand777/test-uv)](https://img.shields.io/github/license/deadhand777/test-uv)

> This is a test of the new project templete for ai show cases.

- **Github repository**: <https://github.com/deadhand777/test-uv/>
- **Documentation** <https://deadhand777.github.io/test-uv/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:deadhand777/test-uv.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

## Set Up Your Development Environment

```bash
uv venv --python <python-version> # e.g. 3.11.6
source .venv/bin/activate
which python
python -V
```

```bash
uv sync
uv pip list
uv add <package>
uv lock
```

## Update project.toml

[tool.ruff]
...

- 'ignore' -> 'lint.ignore'
- 'select' -> 'lint.select'

'per-file-ignores' -> 'lint.per-file-ignores' in [tool.ruff.lint.per-file-ignores]

## Linting & Styling

- Run ruff:

```bash
uv run ruff format
```

- Run pre-commit hooks:

```bash
make check
```

## Testing

```bash
 make test
```

## Update Github Action uv version

- see [setup-uv](https://github.com/astral-sh/setup-uv)

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v3 -- init is uv@v2
  with:
    version: ${{ inputs.uv-version }}
    enable-cache: "true"
    cache-suffix: ${{ matrix.python-version }}
```

## Documentation from MKDocs

- navigate to Settings > Actions > General in your repository, and under Workflow permissions select Read and write permissions

- change the content in `./docs/index.md` or other files

- update `mkdocs.yml`

```bash
make docs
```

- after the changes are pushed to GitHub: create a new release or pre-release to trigger the `pages build and deployment` workflow.

- Settings > Code and Automation > Pages shows the documentatio URL

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Resources

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
