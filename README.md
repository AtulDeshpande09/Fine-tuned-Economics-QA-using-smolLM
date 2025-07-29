# Fine-tuned Economics QA using smolLM

This repo demonstrates a minimal instruction-tuning pipeline for fine-tuning a tiny language model (`smolLM`) using a small, domain-specific dataset (Economics QA).

## Objective
To understand and test the mechanics of instruction tuning on a low-resource, lightweight model using only 55 examples.

## Structure

- `data/sample.jsonl` – 10 example data points for inspection
- `notebook/fine_tuning_pipeline.ipynb` – full training & inference pipeline (runs on Colab)
- `model/` – tokenizer files and config (model weights excluded)

## Notes

- Training done on Google Colab (no paid GPU)
- Model weights are not included to reduce repo size
- Intended as an experiment / proof of concept

## Future Work
A larger version of this project will be built using `flan-t5-small` and ~800 examples.

---

This project serves as a step toward larger instruction-tuning experiments and LLM research.
