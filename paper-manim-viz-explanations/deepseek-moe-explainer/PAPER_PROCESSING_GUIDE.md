# DeepSeek-MoE Paper Processing Guide

A complete guide for processing the DeepSeek-MoE paper into semantic chunks and aligning them with code implementations.

## Overview

This guide covers three main steps:
1. **Extract sections** from the LaTeX paper
2. **Split into semantic chunks** using LLM
3. **Align chunks with code** implementations

## Prerequisites

### 1. Activate Python Environment

```bash
pyenv activate deepseek-moe
```

### 2. Verify NVIDIA API Key

Ensure your `.env` file exists at the project root:

```bash
cat /home/ubuntu/github/grokipedia-research/.env
# Should contain: NVIDIA_API_KEY=your_api_key_here
```

### 3. Install Dependencies (if needed)

```bash
pip install openai python-dotenv httpx
```

---

## Step 1: Extract Sections from LaTeX

Extract specific sections from the paper and convert to Markdown.

```bash
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/ai

python latex_section_extractor.py \
    --file ../deepseek-moe-explainer/deepseek-moe-paper/main.tex \
    --sections "Introduction,Preliminaries: Mixture-of-Experts for Transformers,DeepSeekMoE Architecture" \
    --output-dir ../deepseek-moe-explainer/extracted_sections
```

**Output:**
```
extracted_sections/
├── introduction.md
├── preliminaries_mixture-of-experts_for_transformers.md
└── spmoe_architecture.md
```

---

## Step 2: Split Sections into Semantic Chunks

Use LLM to split each section into semantically coherent chunks.

```bash
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/ai

python semantic_chunker.py \
    --input-dir ../deepseek-moe-explainer/extracted_sections \
    --output-dir ../deepseek-moe-explainer/chunks
```

**Output:**
```
chunks/
├── introduction/
│   ├── _all_chunks.md
│   ├── chunk_01_scaling_language_models_with_moe.md
│   ├── chunk_02_limitations_of_existing_moe_architectures.md
│   └── ...
├── preliminaries_mixture-of-experts_for_transformers/
│   ├── _all_chunks.md
│   ├── chunk_01_standard_transformer_architecture.md
│   └── ...
└── spmoe_architecture/
    ├── _all_chunks.md
    ├── chunk_01_deepseekmoe_overview.md
    ├── chunk_02_fine-grained_expert_segmentation_motivation.md
    └── ...
```

---

## Step 3: Align Chunks with Code Implementations

Map each paper chunk to relevant code sections from the DeepSeek implementation.

```bash
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/deepseek-moe-explainer/aligner

python code_chunk_aligner.py \
    --chunks-dir ../chunks \
    --code-dir ../deepseek-code \
    --output-dir ../chunks-with-code
```

**Output:**
```
chunks-with-code/
├── introduction/
│   ├── chunk_01_scaling_language_models_with_moe.md      # with code
│   └── ...
├── preliminaries_mixture-of-experts_for_transformers/
│   └── ...
└── spmoe_architecture/
    ├── chunk_01_deepseekmoe_overview.md                  # with code
    ├── chunk_06_shared_expert_isolation_motivation.md    # with code
    └── ...
```

Each chunk now contains:
- Original paper content
- Relevance score (0-10)
- Matching code sections with file paths and line numbers

---

## Quick Start: Run All Steps

Run all three steps in sequence:

```bash
# Step 1: Extract sections
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/ai
python latex_section_extractor.py \
    --file ../deepseek-moe-explainer/deepseek-moe-paper/main.tex \
    --sections "Introduction,Preliminaries: Mixture-of-Experts for Transformers,DeepSeekMoE Architecture" \
    --output-dir ../deepseek-moe-explainer/extracted_sections

# Step 2: Create semantic chunks
python semantic_chunker.py \
    --input-dir ../deepseek-moe-explainer/extracted_sections \
    --output-dir ../deepseek-moe-explainer/chunks

# Step 3: Align with code
cd ../deepseek-moe-explainer/aligner
python code_chunk_aligner.py \
    --chunks-dir ../chunks \
    --code-dir ../deepseek-code \
    --output-dir ../chunks-with-code
```

---

## Alternative: Use Pipeline Script

For steps 1 and 2, use the combined pipeline:

```bash
cd /home/ubuntu/github/grokipedia-research/paper-manim-viz-explanations/ai

python paper_chunking_pipeline.py \
    --latex-file ../deepseek-moe-explainer/deepseek-moe-paper/main.tex \
    --sections "Introduction,Preliminaries: Mixture-of-Experts for Transformers,DeepSeekMoE Architecture" \
    --output-dir ../deepseek-moe-explainer

# Then run aligner separately
cd ../deepseek-moe-explainer/aligner
python code_chunk_aligner.py \
    --chunks-dir ../chunks \
    --code-dir ../deepseek-code \
    --output-dir ../chunks-with-code
```

---

## Directory Structure

After running all steps, your directory should look like:

```
deepseek-moe-explainer/
├── aligner/
│   ├── code_chunk_aligner.py     # Aligns chunks with code
│   └── README.md
├── chunks/                        # Semantic chunks (paper only)
│   ├── introduction/
│   ├── preliminaries.../
│   └── spmoe_architecture/
├── chunks-with-code/              # Chunks aligned with code
│   ├── introduction/
│   ├── preliminaries.../
│   └── spmoe_architecture/
├── deepseek-code/                 # Source code files
│   ├── modeling_deepseek.py
│   └── configuration_deepseek.py
├── deepseek-moe-paper/            # Original LaTeX paper
│   └── main.tex
├── extracted_sections/            # Raw extracted sections
│   ├── introduction.md
│   ├── preliminaries_mixture-of-experts_for_transformers.md
│   └── spmoe_architecture.md
└── PAPER_PROCESSING_GUIDE.md      # This file
```

---

## Example Output: Aligned Chunk

Here's what an aligned chunk looks like:

```markdown
# Shared Expert Isolation Motivation

With a conventional routing strategy, tokens assigned to different experts 
may necessitate some common knowledge or information...

---

## Corresponding Code Implementation

**Relevance Score:** 9/10

**Explanation:** The code directly implements the shared expert isolation 
concept described in the paper...

### Code Section 1: DeepseekMoE Class

**File:** `modeling_deepseek.py` (lines 361-374)

```python
class DeepseekMoE(nn.Module):
    """A mixed expert module containing shared experts."""
    def __init__(self, config):
        ...
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekMLP(...)
```
```

---

## Troubleshooting

### "No module named 'openai'"
```bash
pip install openai python-dotenv httpx
```

### "NVIDIA_API_KEY not found"
Create `.env` file at project root:
```bash
echo "NVIDIA_API_KEY=your_key_here" > /home/ubuntu/github/grokipedia-research/.env
```

### "Section not found"
Check exact section names in the LaTeX file. The extractor handles `\spmoe{}` commands automatically.

---

## Scripts Reference

| Script | Location | Purpose |
|--------|----------|---------|
| `latex_section_extractor.py` | `ai/` | Extract sections from LaTeX |
| `semantic_chunker.py` | `ai/` | Split sections into chunks |
| `paper_chunking_pipeline.py` | `ai/` | Combined extract + chunk |
| `code_chunk_aligner.py` | `aligner/` | Align chunks with code |

---

## Model Used

- **LLM:** `qwen/qwen3-coder-480b-a35b-instruct`
- **API:** NVIDIA Build API (`https://integrate.api.nvidia.com/v1`)

