# Release validation

Unit and integration checks for release 0.2.1 passed again on 2026-10-13. Full-data validation was repeated on 2026-10-09 with the fixed `data/` layout. Earlier full-pool input and GT-isolation evidence is retained in `metadata/validation.json`.

| Check | Result |
| --- | --- |
| Core unit/regression tests | 44 passed on 2026-10-13 |
| Shell workflow integration tests | 9 passed on 2026-10-13 |
| Local deployment / native-server tests | 12 passed on 2026-10-13 |
| Fixed data directory and archive round-trip tests | 2 passed on 2026-10-13 |
| Total current unit/integration tests | 67 passed |
| Public question records | 4,467; no `source` field |
| Media references | All relative `images/` paths |
| Media files | 42,637 regular files; every image decoded and SHA-256 checked |
| Media bytes | 4,651,532,358 |
| Prepared corpus | All 4,467 questions compiled; references isolated in the scoring file |
| GT isolation | All 4,467 public inputs unchanged after replacing the reference text with a sentinel |
| Relocation | Dataset, prepared corpus and partial run moved together; resume and scoring passed |
| Real-input local mock | 14 solve requests, 120 images, 2 open-task judge requests |
| Real model requests during validation | 0 |
| Binary judge protocol | Both correct/1 and incorrect/0 exercised; invalid JSON judgments rejected |
| Protocol validation | Missing prompt protocol and mismatched judge protocol rejected |

The evaluator was built as a wheel and installed into a separate virtual environment. The installed CLI resolved the fixed dataset from a checkout subdirectory and prepared one real input per task: 14 questions and 120 images. Shell tests invoked the checkout from another working directory and verified that a different dataset in that working directory could not redirect the evaluation. These checks verify packaging and execution without measuring model or judge accuracy.

Machine-readable evidence is in `metadata/validation.json`, `metadata/data_layout_validation.json` and `metadata/deployment_validation.json`. The media checksum list is `metadata/images.jsonl`. Reproduce the unit tests, full dataset validation and local mock run using the commands in [Evaluation](EVALUATION.md).

The Shell entry point was checked against the real CLI and a localhost mock service. Its nine integration tests cover complete execution and resume, paths containing spaces, execution from another directory, incomplete inference and explicit retry, judge configuration validation before inference, changed corpus selection, closed-question scoring, incomplete judge results without resampling, custom solver prompts with resume checks, missing-download diagnostics, and rejection of dataset-path overrides by both entry points. No real model requests were made.

Archive tests verify that the code ZIP contains only `data/.gitkeep`, that the data archive contains only the question JSONL and `images/`, and that extracting the latter into the former's `data/` enables preparation and checksum validation. Validation uses the repository's `metadata/` without requiring downloaded metadata. The complete local dataset was moved without changing its JSONL or image bytes; all 4,467 records and 42,637 image hashes/decodes passed again. See `metadata/data_layout_validation.json`.

Local deployment checks cover all 24 profiles against the current client schema, exact generation/reasoning settings from the paper, image limits and checkpoint-relative Phi LoRA paths, both HY template modes, the InternVL answer boundary, and HY native-server HTTP authentication, generation serialization and overflow rejection using mock tensor/model computation. The normal launch path was exercised with a mocked runtime probe and process execution for both vLLM and HY MoT, without supplying a weight revision. Checks cover generated model/judge configs, distinct deployment identifiers and refusal to overwrite deployment records. Evaluator tests cover inference and resume with the optional version omitted, and rejection of a changed declared version on resume. Environment-only authentication and all five installation plans were also checked during the 2026-10-09 deployment validation.

Both compatibility patches were applied to copies of their exact upstream source files and verified against the expected resulting hashes; reapplying them was idempotent. Phi crop padding was checked numerically with CPU tensors. New deployment/config files were scanned for private machine paths and internal service/job identifiers. The code archive includes the deployment assets and excludes evaluation media, weights, environments and generated outputs.

These deployment checks do not build CUDA extensions, load real weights or reproduce model accuracy. See `metadata/deployment_validation.json` for the scope and the [deployment guide](../deploy/README.md) for the scope of deployment records.
