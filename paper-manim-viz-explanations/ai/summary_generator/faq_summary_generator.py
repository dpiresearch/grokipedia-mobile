#!/usr/bin/env python3
"""
FAQ Summary Generator

This script takes a FAQ markdown file (from Cursor conversations) and generates
a comprehensive, coherent summary of ALL FAQs together that includes:
1. Clear separation by FAQ number
2. Mathematical explanations and intuitions
3. Coherent narrative flow since questions are often sequentially linked
4. Code examples and implementation details
5. Key takeaways and connections between concepts

Uses NVIDIA Nemotron API with Qwen Coder 3 480B model.
"""

import argparse
import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "qwen/qwen3-coder-480b-a35b-instruct"

# System prompt for generating comprehensive FAQ summaries
SYSTEM_PROMPT = """You are an expert technical writer and educator specializing in machine learning, deep learning, and transformer architectures. Your task is to create a comprehensive, educational summary document that synthesizes a series of related FAQs into a coherent learning resource.

The FAQs you'll receive are from a Q&A session about DeepSeek MoE (Mixture of Experts) model implementation. The questions often build upon each other, forming a natural learning progression.

Your output should be a SINGLE, cohesive document that:

## Document Structure

### 1. Executive Overview
- Briefly summarize what the entire FAQ session covers
- Identify the main themes and learning objectives
- Explain how the questions connect to form a learning journey

### 2. FAQ Deep Dives (for EACH FAQ)
For each FAQ, create a clearly numbered section with:

#### a) The Question (rephrased for clarity if needed)
#### b) Core Answer & Intuition
- Explain the concept in accessible terms
- Provide analogies or visualizations when helpful
- Connect to prior FAQs to maintain narrative flow

#### c) Technical Deep Dive
- Mathematical formulations with step-by-step explanations
- Use LaTeX notation: $inline$ and $$block$$ equations
- Explain every variable and operation
- Show derivations where applicable

#### d) Code Implementation
- Explain the relevant code snippets
- Highlight key functions and their purpose
- Show how theory maps to implementation

#### e) Worked Example (when applicable)
- Concrete numerical examples
- Step-by-step computation with actual values
- Expected outputs and interpretation

### 3. Connections & Synthesis
- How do these concepts relate to each other?
- What's the bigger picture?
- Common misconceptions to avoid

### 4. Key Takeaways
- Summary of the most important points
- Practical implications
- Further reading suggestions

## Formatting Guidelines
- Use proper Markdown with headers (##, ###, ####)
- Format code with ```python blocks
- Use LaTeX for math: $inline$ and $$block$$
- Use bullet points and numbered lists for clarity
- Include horizontal rules (---) between major sections
- Make it scannable but also deep

## Important Notes
- MAINTAIN COHERENCE: Questions often reference previous ones
- ADD MATHEMATICAL RIGOR: Expand on mathematical explanations
- PROVIDE INTUITION: Explain the "why" not just the "what"
- BE THOROUGH: This is a reference document, not a summary

Your goal is to transform a Q&A session into a professional, educational document that someone could use to deeply understand DeepSeek MoE implementation."""


def parse_faq_file(filepath: str) -> List[Dict]:
    """
    Parse a FAQ markdown file and extract Q&A pairs.
    
    Returns list of dicts with 'question', 'answer', and 'index' keys.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by the separator ---
    sections = re.split(r'\n---\n', content)
    
    faqs = []
    current_question = None
    current_answer = []
    faq_index = 0
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check if this is a User section (question)
        if section.startswith('**User**'):
            # Save previous Q&A if exists
            if current_question and current_answer:
                faq_index += 1
                faqs.append({
                    'index': faq_index,
                    'question': current_question.strip(),
                    'answer': '\n'.join(current_answer).strip()
                })
            
            # Extract new question
            current_question = section.replace('**User**', '').strip()
            current_answer = []
            
        elif section.startswith('**Cursor**'):
            # This is an answer
            answer_text = section.replace('**Cursor**', '').strip()
            current_answer.append(answer_text)
    
    # Don't forget the last Q&A pair
    if current_question and current_answer:
        faq_index += 1
        faqs.append({
            'index': faq_index,
            'question': current_question.strip(),
            'answer': '\n'.join(current_answer).strip()
        })
    
    return faqs


def create_faq_title(question: str, max_length: int = 80) -> str:
    """Create a concise title from the question."""
    # Remove markdown formatting
    title = re.sub(r'@\S+', '', question)  # Remove @mentions
    title = re.sub(r'`[^`]+`', '', title)  # Remove code blocks
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Take first line if multi-line
    title = title.split('\n')[0]
    
    # Truncate if needed
    if len(title) > max_length:
        title = title[:max_length-3] + "..."
    
    return title if title else "FAQ"


def create_summary_prompt(faqs: List[Dict], source_filename: str) -> str:
    """Create the prompt for generating the comprehensive FAQ summary."""
    
    # Build FAQ content string
    faq_content_parts = []
    for faq in faqs:
        faq_title = create_faq_title(faq['question'])
        faq_content_parts.append(f"""
---

## FAQ #{faq['index']}: {faq_title}

**Full Question:**
{faq['question']}

**Detailed Answer:**
{faq['answer']}
""")
    
    faq_content = '\n'.join(faq_content_parts)
    
    return f"""Please create a comprehensive, coherent summary document synthesizing the following {len(faqs)} FAQs from a Q&A session about DeepSeek MoE (Mixture of Experts) model implementation.

**Source:** {source_filename}
**Total FAQs:** {len(faqs)}

These FAQs form a natural learning progression - the questioner starts with basic questions about the forward pass, then dives into context length, RoPE embeddings, and implementation details. Your summary should maintain this narrative flow while adding mathematical rigor and educational depth.

---

# THE FAQs TO SUMMARIZE:

{faq_content}

---

# YOUR TASK:

Create a comprehensive educational document following the structure in your instructions:
1. Executive Overview (themes, connections, learning objectives)
2. Deep Dive for EACH FAQ (question, intuition, math, code, examples)
3. Connections & Synthesis
4. Key Takeaways

IMPORTANT:
- Maintain narrative coherence - later questions reference earlier ones
- Add mathematical explanations where the original answer was brief
- Provide intuition for complex concepts
- Include worked examples with actual numbers
- Use proper Markdown formatting throughout
- Make it suitable as a standalone reference document

Generate the complete summary document now."""


async def generate_summary(prompt: str) -> str:
    """Generate summary using NVIDIA Nemotron API with Qwen Coder."""
    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY
    )
    
    print("  Generating comprehensive FAQ summary with Qwen Coder...")
    print("  (This may take a while for detailed content...)")
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,  # Large output for comprehensive summary
    )
    
    return response.choices[0].message.content


def clean_summary(summary: str) -> str:
    """Clean up the generated summary."""
    # Remove thinking tags if present
    summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
    
    # Remove markdown code block wrappers if the entire response is wrapped
    if summary.strip().startswith('```markdown'):
        summary = summary.strip()[len('```markdown'):].strip()
        if summary.endswith('```'):
            summary = summary[:-3].strip()
    elif summary.strip().startswith('```'):
        summary = summary.strip()[3:].strip()
        if summary.endswith('```'):
            summary = summary[:-3].strip()
    
    return summary.strip()


def save_summary(summary: str, output_dir: Path, filename: str) -> str:
    """Save the generated summary to a markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"  Summary saved to: {filepath}")
    return str(filepath)


async def generate_faq_summary(
    faq_file: str,
    output_dir: str,
    output_filename: str = None
) -> Dict:
    """
    Generate a comprehensive summary of all FAQs.
    
    Args:
        faq_file: Path to the FAQ markdown file
        output_dir: Directory for output
        output_filename: Optional custom output filename
    
    Returns:
        Dictionary with results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("FAQ SUMMARY GENERATOR")
    print(f"{'='*70}")
    print(f"\nFAQ file: {faq_file}")
    print(f"Output directory: {output_dir}")
    
    # Parse FAQs
    faqs = parse_faq_file(faq_file)
    
    print(f"\nFound {len(faqs)} FAQ(s):")
    for faq in faqs:
        title = create_faq_title(faq['question'], max_length=60)
        print(f"  {faq['index']}. {title}")
    
    # Generate the output filename
    if not output_filename:
        source_name = Path(faq_file).stem
        output_filename = f"{source_name}_comprehensive_summary.md"
    
    print(f"\nOutput file: {output_filename}")
    
    try:
        # Generate comprehensive summary
        source_filename = Path(faq_file).name
        prompt = create_summary_prompt(faqs, source_filename)
        
        print(f"\n{'='*60}")
        print("Generating Comprehensive Summary")
        print(f"{'='*60}")
        
        summary = await generate_summary(prompt)
        
        # Clean up the summary
        summary = clean_summary(summary)
        
        # Add header with metadata
        header = f"""# DeepSeek MoE FAQ Comprehensive Summary

> **Source:** {source_filename}  
> **FAQs Covered:** {len(faqs)}  
> **Generated:** Comprehensive educational summary with mathematical explanations and intuitions

---

"""
        full_summary = header + summary
        
        # Save the summary
        summary_filepath = save_summary(full_summary, output_path, output_filename)
        
        print(f"\n  ✓ SUCCESS! Comprehensive summary generated")
        print(f"    FAQs summarized: {len(faqs)}")
        print(f"    Output: {summary_filepath}")
        
        return {
            'faq_file': faq_file,
            'faq_count': len(faqs),
            'summary_file': summary_filepath,
            'success': True
        }
        
    except Exception as e:
        print(f"\n  ✗ FAILED: {type(e).__name__}")
        print(f"    Error: {str(e)[:500]}")
        
        return {
            'faq_file': faq_file,
            'faq_count': len(faqs) if faqs else 0,
            'summary_file': None,
            'success': False,
            'error': str(e)
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive educational summary from FAQ file"
    )
    parser.add_argument(
        "--faq-file", "-f",
        type=str,
        required=True,
        help="Path to the FAQ markdown file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for generated summary"
    )
    parser.add_argument(
        "--output-filename", "-n",
        type=str,
        default=None,
        help="Custom output filename (default: <source>_comprehensive_summary.md)"
    )
    
    args = parser.parse_args()
    
    result = await generate_faq_summary(
        faq_file=args.faq_file,
        output_dir=args.output_dir,
        output_filename=args.output_filename
    )
    
    # Print final result
    print(f"\n\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    
    if result['success']:
        print(f"\n✓ Successfully generated comprehensive FAQ summary!")
        print(f"  FAQs covered: {result['faq_count']}")
        print(f"  Output: {result['summary_file']}")
    else:
        print(f"\n✗ Failed to generate summary")
        print(f"  Error: {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    asyncio.run(main())

