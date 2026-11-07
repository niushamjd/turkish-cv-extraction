# turkish-cv-extraction

Layout-aware information extraction from Turkish CVs (Document AI seminar, Uni Heidelberg)

Seminar project by Niyousha Mojoudi and İpek Uzun. Follows Zhu et al. (2025), arXiv:2510.09722, adapted to Turkish.

## Structure
- `prompts.py`: the three extraction prompts (basic info + skills + languages, education, work experience) in the paper's instruction format, plus helpers: line indexing, SFT sample builder, JSON parsing, pointer resolution.

## Roadmap
1. [x] Schema and prompts
2. [ ] Synthetic CV generator: LLM makes JSON profiles, HTML templates render them to PDF, the JSON is the label
3. [ ] Text extraction (PyMuPDF text layer + OCR for image-only PDFs) and reading-order step
4. [ ] Baselines 1 to 3 (Tesseract + rules, naive LLM, VLM)
5. [ ] Evaluation: Hungarian alignment + per-field matching, Turkish-aware lowercasing
6. [ ] LoRA fine-tuning of Qwen3-0.6B with MLX
7. [ ] RAG search and demo app
