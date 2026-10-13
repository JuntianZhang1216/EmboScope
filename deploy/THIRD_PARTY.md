# Third-party components

Model weights and upstream packages are downloaded separately and retain their own licenses. This directory does not vendor complete vLLM, Transformers or HY plugin implementations.

- `patches/phi4mm-crops.patch` modifies vLLM's `vllm/model_executor/models/phi4mm.py` at commit `c21d0039ecc85b06617034ff2166a1fb79309d53`. Copyright contributors to the vLLM project; Apache License 2.0. The patch adds joint zero-padding of image crops and masks.
- `patches/hy-vlm10-vllm.patch` modifies the HY-Embodied vLLM plugin at commit `16289e99111ee362a23f93d5b3a463ddc844decc`. The patched modules carry Apache-2.0 notices crediting the Hunyuan and vLLM teams and their upstream contributors. The patch adapts vLLM APIs and preserves the non-thinking template flag.
- `templates/hy-embodied-vlm-10.jinja` adapts the HY-VLM chat serialization for the benchmark's OpenAI image content blocks and system instruction. Its model-specific tokens remain governed by the upstream model distribution.

A copy of the [Apache License 2.0](licenses/Apache-2.0.txt) accompanies the patch material. Installing these components does not grant a license to redistribute their model weights. The evaluation code uses the root [Apache License 2.0](../LICENSE); dataset licensing is separate.
