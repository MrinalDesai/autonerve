# Local models

Place model weights here. AutoNerve uses Qwen2.5-1.5B-Instruct (same as the UPS app).

Expected layout:
    backend/models/Qwen2.5-1.5B-Instruct/
        config.json
        model.safetensors        (or pytorch_model.bin)
        tokenizer.json
        tokenizer_config.json
        ... (all files from the HF snapshot)

How to get it (pick one):
  1) Copy the folder you already have from the UPS app into here.
  2) Download once:
       pip install huggingface_hub
       huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
         --local-dir backend/models/Qwen2.5-1.5B-Instruct

llm.py resolves the model in this order:
  - env var AUTONERVE_MODEL  (absolute path), else
  - backend/models/Qwen2.5-1.5B-Instruct  (this folder, the default)

Until weights are present, llm.available() is False and extraction.py uses the
deterministic keyword fallback, so the full thread still runs.

Weights are large — do NOT commit them. See .gitignore.
