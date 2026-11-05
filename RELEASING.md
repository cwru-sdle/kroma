# Releasing KROMA

A release publishes `kroma-om` from the latest `main`.

```bash
gh workflow run release.yml -R cwru-sdle/kroma -f bump=patch
```

Or in GitHub: **Actions → Release → Run workflow**.

| Bump | Use it for | Example |
|:--|:--|:--|
| `patch` | Bug fixes | 0.3.0 → 0.3.1 |
| `minor` | New features | 0.3.0 → 0.4.0 |
| `major` | Changes that break existing use | 0.3.0 → 1.0.0 |

The workflow runs the tests on `main`, bumps the version, builds and checks the package, commits, tags
`vX.Y.Z`, creates the GitHub release with generated notes, and starts `publish.yml`, which tests the tag
again and uploads to PyPI with trusted publishing. There is no API token to manage.

It stops without changing anything if the tests or the build fail, if `main` has no commits since the
last tag, or if the tag already exists.

**A PyPI version can never be reused**, even if the release is deleted. A bad release is fixed by a new one.

## Before you release

1. Merge everything the release should contain. It takes whatever is on `main`.
2. For a change to the benchmark pipeline, run `kroma run` on a real track. CI only tests it with a
   scripted model.

To see what would happen first, add `-f dry_run=true`. It runs everything up to the commit, then stops.

## After you release

```bash
gh run list -R cwru-sdle/kroma --workflow publish.yml --limit 1
pip install --upgrade kroma-om && python -c "import kroma; print(kroma.__version__)"
```

PyPI can take a few minutes to show the new version.

## If something goes wrong

- **The release workflow failed before "Commit, tag":** nothing changed. Fix it and run it again.
- **The upload failed after the tag was pushed:** the version isn't on PyPI yet, so fix the cause and rerun
  the upload: `gh workflow run publish.yml -R cwru-sdle/kroma --ref vX.Y.Z -f target=pypi`.
- **`main` becomes protected:** the workflow pushes the version commit straight to `main`, so its bot needs to
  be allowed to bypass the rule.

## Setup this relies on

PyPI and TestPyPI each need a trusted publisher for `cwru-sdle/kroma`, workflow `publish.yml`, and environment
`pypi` or `testpypi`. PyPI's is already in place, because `v0.2.0` and `v0.3.0` published this way.
Read the Docs rebuilds through its own GitHub webhook, separately from this.
