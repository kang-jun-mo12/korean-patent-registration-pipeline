# H200 LoRA Adapter + SBERT + 7B Subclass Classifier Training Code Bundle

This bundle contains the H200 LoRA adapter training code, the SBERT H200 training code, and the 7B subclass classifier training code copied from the handoff workspace.

Included:
- `lora_adapters/h200_bf16_lora`: H200 BF16 LoRA adapter training, smoke run, and evaluation scripts.
- `sbert_h200`: SBERT H200 training, indexing, retrieval, and recall evaluation scripts.
- `subclass_classifier_7b`: Qwen2.5 7B subclass classifier LoRA training and evaluation scripts.

Excluded intentionally:
- PRO6000 adapter code.
- QLoRA taskmix code.
- Large JSONL datasets.
- Raw zip files.
- Model checkpoints and adapter weights.
- `__pycache__` and generated cache files.

Copy the required data files back into each package's expected `data` or project root paths before running training scripts.
