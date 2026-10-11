# HCMUS_Beyonders at AstroCLIMB (WASP 2026 shared task)

Code for our system description paper, *HCMUS_Beyonders at AstroCLIMB: Same Paper or Citing Paper?
Linking Astronomy Figures with OCR, Citation Graphs, and LLMs*.

The task: given two objects taken from astronomy papers, each a figure or a caption, decide whether they
are the same figure, come from the same paper, come from papers linked by a citation, or are unrelated.

## Layout

| Path | Contents |
|---|---|
| `src/` | The pipeline and every experiment reported in the paper |
| `jobs/` | Shell scripts that record the exact settings of each run |
| `scripts/diag/` | One-off diagnostic scripts used for the error analysis |

Comments and log messages in the code are written in Vietnamese.

## Setup

Python 3.10. Install PyTorch for your CUDA version first (we used `torch==2.13.0` with CUDA 12.6), then:

```bash
pip install -r requirements.txt
```

Each LLM classifier needs a 40 GB GPU.

## Data

No data is included. Download `train.csv`, `test.csv` and `sample_submission.csv` from the
[Kaggle competition](https://www.kaggle.com/competitions/astroclimb) and the public corpus from
[adsabs/AstroCLIMB](https://huggingface.co/datasets/adsabs/AstroCLIMB).

Every script reads its paths from the environment variable `WASP_ROOT`, a directory that contains
`data/` (the CSV files) and `outputs/` (everything the pipeline writes). The default value in the code is
the path on our cluster, so set it before running anything:

```bash
export WASP_ROOT=/path/to/workdir
export PY=python          # interpreter used by the scripts in jobs/
```

## Pipeline

Each step writes to `$WASP_ROOT/outputs` and no step except the first reads the original CSV files again.

| Step | Script | What it does |
|---|---|---|
| 1 | `src/embed.py` | One pass over the CSV files: decode images, cache them, compute frozen embeddings |
| 2 | `src/ocr.py`, `src/ocr_tiles.py` | Read the text inside each figure (whole image, or tiles) |
| 3 | `src/embed_ocr.py` | Embed the OCR text |
| 4 | `src/graph_fetch.py`, `src/pairs.py`, `src/ocr_hf.py` | Build the citation-graph pretraining pairs from the public corpus, with real OCR text |
| 5 | `src/train_ce.py` | Pretrain and fine-tune one cross-encoder with five-fold cross-validation |
| 6 | `src/ce_ens.py` | Average the cross-encoders into the ensemble |
| 7 | `src/features.py`, `src/proj_nce.py`, `src/baseline.py` | Similarity and rank features, and the stacker |
| 8 | `src/llm_lora.py` | Fine-tune an LLM classifier with LoRA (Qwen3-8B, Gemma-4-12B) |
| 9 | `src/blend_stack.py`, `src/blend_llm2.py` | Blend the stacker with one or two LLMs |
| 10 | `src/submit.py` | Check the submission format |

The settings of each run in the paper are in `jobs/`: for example `jobs/lora_full.sh` and
`jobs/lora_gemma4.sh` for the two submitted LLMs.
