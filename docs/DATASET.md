# Dataset contract

The dataset contains 4,467 synthetically constructed visual question-answer records. The release uses no per-record `source` field.

Download the JSONL and media into the code checkout's fixed `data/` directory: `data/benchmark_data.jsonl` and `data/images/`. Git and the code archive contain only `data/.gitkeep`; the checksum manifests are included separately in the repository-root `metadata/`. Evaluation requires no dataset-path configuration.

| Field | Meaning |
| --- | --- |
| `id` | Stable `q000001`–`q004467` identifier in dataset order |
| `task` | One of the 14 names in the packaged `task_catalog.json`; metadata, not a prompt |
| `text_list` | Text segments; the final segment is the reference answer |
| `image_list` | Ordered images with relative paths, dimensions, checksums and text anchors |
| `role_list` | Conversation roles for the text and image entries |

Each media entry has `image`, `matched_text_index`, `width`, `height` and `sha256`. Images are inserted **before** the text segment at `matched_text_index`. Images anchored before the final answer segment are still public observations; the answer text itself is omitted from model inputs. Frame sequence entries also have `source_frame_index`, `pts_sec` and `text_before`, describing frame positions, timestamps and text displayed before each frame.

Every image path is `images/<full-sha256>.<jpg|png|webp>` relative to the JSONL directory. The evaluator sends the encoded image bytes at their supplied dimensions and in their specified order. The directory contains regular files, not symlinks or dependencies on external mounts. Identical image bytes share a file: there are 42,637 files for 42,643 references.

The 229 adjacent-action questions use fixed ordered frame sequences. `metadata/video_frames.jsonl` records per-question frame indices and presentation timestamps. All frames are supplied in `images/`; evaluation needs no video decoder.

| Task | Level | Questions | Answer format |
| --- | --- | ---: | --- |
| relative_distance | L1 | 300 | single_choice |
| attribute_matching | L1 | 289 | single_choice |
| cross_view_matching | L1 | 300 | single_choice |
| frame_completion | L1 | 352 | single_choice |
| action_frame_matching | L2 | 217 | single_choice |
| action_between_frames | L2 | 331 | single_choice |
| step_frame_alignment | L2 | 257 | single_choice |
| action_sequence_description | L2 | 218 | open |
| adjacent_action | L2 | 229 | open |
| step_completion | L2 | 500 | yes_no |
| action_prediction_selection | L3 | 557 | single_choice |
| progress_comparison | L3 | 293 | single_choice |
| progress_ordering | L3 | 330 | single_choice |
| progress_anomaly_detection | L3 | 294 | single_choice |

All questions are released as one evaluation pool. No random train/test split is implied. Reference answers are included; users assessing generalization should avoid training on this evaluation pool. Inclusion does not by itself indicate independent human verification of every reference answer.

Dataset, prepared inputs and run configurations are identified by checksums. Evaluation and scoring require matching input hashes.
