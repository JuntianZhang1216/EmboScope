# Packaging the local release

The Git repository and code archive contain code, documentation, checksum metadata and an empty `data/` placeholder. The question JSONL and evaluation media are distributed together in a separate dataset archive. README paper figures remain in `docs/figures/` and are included with the code.

The local publisher prepares `data/benchmark_data.jsonl` and `data/images/`, installs the evaluator, and validates the data from the checkout:

```bash
embodied-eval validate --output artifacts/data_validation.json
python eval/scripts/pack_release.py --output ../Embodied_Benchmark_dist/fixed-data
```

Use a new output directory. This creates:

- `embodied-benchmark-code-v0.2.1.zip`: `evaluate.sh`, the evaluator, local deployment code and compatibility assets, 24 model configurations, tests, documentation, README figures, checksum metadata and `data/.gitkeep`. It contains no question JSONL or evaluation images.
- `embodied-benchmark-data-v0.2.1.tar.gz`: all 4,467 records as `benchmark_data.jsonl`, plus all 42,637 images under `images/`.
- `SHA256SUMS-v0.2.1`: hashes of both archives.

Clone or extract the code repository first. From its root, extract the matching data archive into the existing `data/` directory:

```bash
tar -xzf /path/to/embodied-benchmark-data-v0.2.1.tar.gz -C data
```

The resulting paths must be `data/benchmark_data.jsonl` and `data/images/`; do not create `data/data/`. Users downloading the JSONL and media separately should populate the same paths. The repository-root `metadata/` already contains the validation manifests. Dataset-path arguments are not needed or accepted by the evaluation entry points.

Archive paths are relative and no symlinks are packaged. The data archive strips local usernames, numeric owner IDs and original timestamps from tar headers. Do not include generated outputs, local API configurations, credentials, model weights or machine environments. The code packer explicitly includes only `data/.gitkeep` from the data directory, even when the publisher has populated it locally.

For a code-only update, use a new output directory and add `--code-only`. This writes the code ZIP and a `.sha256` sidecar without reading or rebuilding the media archive.

## Optional GitHub Actions

GitHub Actions is disabled in this release. The test workflow is provided as a template at [`github-actions/tests.yml`](github-actions/tests.yml). Local tests remain available with `python -m unittest discover -s eval/tests -v` after installing `.[test]`.

To enable automatic tests later, copy the template to `.github/workflows/tests.yml` and commit it using credentials that permit workflow updates.
