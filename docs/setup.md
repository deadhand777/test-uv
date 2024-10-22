## Getting started with your project

![](./man/setup.jpeg)

### 1. Create a New Repository with this template

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

- add the following to `project.toml` if additional folders are present in the root directory:

```toml
[tool.setuptools]
py-modules = ["test_uv", "cdk", "additional_modules"]
```

- modify the following to `project.toml` to avoid deprecation warnings:

```toml
[tool.ruff]

'ignore' -> 'lint.ignore'
'select' -> 'lint.select'

'per-file-ignores' -> 'lint.per-file-ignores' in [tool.ruff.lint.per-file-ignores]
```

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

- further information on automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).

## Code Coverage

- To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

- Extend action.yml:

```yaml
- name: Upload results to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
```

## Releasing a new version

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
