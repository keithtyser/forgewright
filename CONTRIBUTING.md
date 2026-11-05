# Contributing

Use Python 3.11–3.13 and Node 24 for the control plane. GPU execution requires a separate Linux host setup; CI does not train models.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps -e .
python -m pytest
python -m ruff check --config pyproject.toml forgewright
python -m mypy forgewright
python -m build
cd frontend
npm ci --ignore-scripts
npm test
npm audit
```

For dependency updates, run `uv pip compile pyproject.toml --extra dev --universal --generate-hashes --output-file requirements-dev.lock`, install the lock in a fresh environment, and rerun CI and `python -m pip_audit`. Commit the Python and npm lock files. Do not remove compatibility pins just to obtain the newest version.

For each behavior change, test the failure path as well as success. Use temporary files and fake external services. Never put credentials, model weights, or real training data in a test fixture.

A GPU smoke run needs an explicit host, device allocation, image digest, model revision, data hashes, and runtime limit. Record the commands, exit status, output manifests, evaluation counts, and model identities. Keep that evidence separate from mocked CI results.

Before a release, require CI on the exact commit and a recorded GPU smoke for each claimed execution path. Check that the Python and frontend versions agree, update the changelog, and then create a version tag. A successful package build alone is not release validation.
