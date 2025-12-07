#!/usr/bin/env python3
"""
Chunk Summary Generator

This script takes aligned chunks-with-code and generates detailed summaries that include:
1. Intuition of the idea explained in the paper
2. How it is practically implemented in the code
3. Small examples with associated code and expected output
4. Explanation of the math behind the output

Uses NVIDIA Nemotron API with Qwen Coder 3 480B model.
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "qwen/qwen3-coder-480b-a35b-instruct"

# System prompt for generating detailed summaries
SYSTEM_PROMPT = """You are an expert technical writer and educator specializing in machine learning and deep learning concepts. Your task is to create comprehensive, educational summaries that help readers deeply understand both the theory and practical implementation of ML concepts.

When given a chunk of text from a research paper along with its corresponding code implementation, you must produce a detailed summary that includes:

## 1. Intuition & Core Idea
- Explain the key insight or motivation behind this concept in simple, accessible terms
- Use analogies or real-world comparisons when helpful
- Explain WHY this approach is needed and what problem it solves

## 2. Technical Deep Dive
- Break down the mathematical formulation step by step
- Explain each variable, parameter, and operation
- Connect the math to the intuition

## 3. Code Implementation Walkthrough
- Explain how the code implements the theoretical concept
- Highlight key functions, classes, and their roles
- Point out important implementation details and design choices

## 4. Worked Example
- Provide a concrete, runnable example with specific input values
- Show the step-by-step computation with actual numbers
- Explain what each intermediate result means
- Show the expected output and interpret its meaning

## 5. Mathematical Derivation (if applicable)
- Show the derivation of key formulas
- Explain the mathematical reasoning
- Connect back to the practical implications

## 6. Key Takeaways
- Summarize the most important points
- Highlight common pitfalls or misconceptions
- Suggest further reading or related concepts

Format your response in clean Markdown with proper headings, code blocks, and mathematical notation (using LaTeX syntax with $ for inline and $$ for block equations).

Be thorough, educational, and precise. Your goal is to help someone truly understand this concept well enough to implement and modify it themselves."""


def create_summary_prompt(chunk_content: str, chunk_title: str) -> str:
    """Create the prompt for generating the detailed summary."""
    return f"""Please create a comprehensive educational summary for the following concept from a research paper on Mixture of Experts (MoE) architectures.

**Topic: {chunk_title}**

Here is the content from the paper along with the corresponding code implementation:

---
{chunk_content}
---

Generate a detailed summary following the structure outlined in your instructions. Make sure to:
1. Start with an intuitive explanation that anyone can understand
2. Dive deep into the technical details and math
3. Walk through the code implementation
4. Provide a concrete worked example with actual numbers
5. Summarize key takeaways

Remember to use proper Markdown formatting with code blocks and LaTeX math notation."""


async def generate_summary(prompt: str) -> str:
    """Generate summary using NVIDIA Nemotron API with Qwen Coder."""
    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY
    )
    
    print("  Generating summary with Qwen Coder...")
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=8000,
    )
    
    return response.choices[0].message.content


def read_chunk_file(filepath: str) -> tuple[str, str]:
    """Read a chunk file and extract title and content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to extract title from first heading
    title_match = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
    else:
        # Use filename as title
        title = Path(filepath).stem.replace('_', ' ').title()
    
    return title, content


def save_summary(summary: str, output_path: Path, chunk_name: str) -> str:
    """Save the generated summary to a markdown file."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{chunk_name}_summary.md"
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"  Summary saved to: {filepath}")
    return str(filepath)


async def process_chunk(
    chunk_file: str,
    output_dir: Path,
    section_name: str = None
) -> Dict:
    """
    Process a single chunk and generate summary.
    
    Args:
        chunk_file: Path to the chunk markdown file
        output_dir: Base directory for output
        section_name: Name of the section (e.g., 'introduction', 'spmoe_architecture')
    
    Returns:
        Dictionary with results
    """
    chunk_title, chunk_content = read_chunk_file(chunk_file)
    chunk_name = Path(chunk_file).stem
    
    # Create output directory structure matching chunks-with-code
    if section_name:
        chunk_output_dir = output_dir / section_name
    else:
        chunk_output_dir = output_dir
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {chunk_title}")
    print(f"Output dir: {chunk_output_dir}")
    print(f"{'='*60}")
    
    try:
        # Generate summary
        prompt = create_summary_prompt(chunk_content, chunk_title)
        summary = await generate_summary(prompt)
        
        # Clean up the response (remove thinking tags if present)
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
        
        # Save the summary
        summary_filepath = save_summary(summary, chunk_output_dir, chunk_name)
        
        print(f"\n  ✓ SUCCESS! Summary generated")
        
        return {
            'chunk_file': chunk_file,
            'chunk_title': chunk_title,
            'summary_file': summary_filepath,
            'success': True
        }
        
    except Exception as e:
        print(f"\n  ✗ FAILED: {type(e).__name__}")
        print(f"    Error: {str(e)[:500]}")
        
        return {
            'chunk_file': chunk_file,
            'chunk_title': chunk_title,
            'summary_file': None,
            'success': False,
            'error': str(e)
        }


async def process_chunks_directory(
    chunks_dir: str,
    output_dir: str,
    limit: int = None
) -> List[Dict]:
    """
    Process all chunks in a directory, organizing output by section.
    
    Args:
        chunks_dir: Directory containing chunk markdown files (organized by section)
        output_dir: Directory for output (will mirror section structure)
        limit: Maximum number of chunks to process (None = all)
    
    Returns:
        List of result dictionaries
    """
    chunks_path = Path(chunks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("CHUNK SUMMARY GENERATOR")
    print(f"{'='*70}")
    print(f"\nChunks directory: {chunks_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all chunk files with their section names
    chunk_entries = []  # List of (section_name, chunk_file_path)
    for section_dir in sorted(chunks_path.iterdir()):
        if section_dir.is_dir():
            section_name = section_dir.name
            for chunk_file in sorted(section_dir.glob('chunk_*.md')):
                chunk_entries.append((section_name, str(chunk_file)))
    
    if limit:
        chunk_entries = chunk_entries[:limit]
    
    print(f"\nFound {len(chunk_entries)} chunk files to process")
    
    # Show sections
    sections = set(entry[0] for entry in chunk_entries)
    print(f"Sections: {', '.join(sorted(sections))}")
    
    # Process each chunk
    results = []
    for i, (section_name, chunk_file) in enumerate(chunk_entries, 1):
        print(f"\n[{i}/{len(chunk_entries)}]", end="")
        result = await process_chunk(
            chunk_file=chunk_file,
            output_dir=output_path,
            section_name=section_name
        )
        results.append(result)
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nTotal: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully generated summaries:")
        for r in successful:
            print(f"  - {r['chunk_title']}")
            print(f"    File: {r['summary_file']}")
    
    if failed:
        print(f"\n✗ Failed chunks:")
        for r in failed:
            print(f"  - {r['chunk_title']}")
            print(f"    Error: {r.get('error', 'Unknown')[:100]}")
    
    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed educational summaries from aligned paper chunks"
    )
    parser.add_argument(
        "--chunks-dir", "-d",
        type=str,
        default=None,
        help="Directory containing aligned chunk files"
    )
    parser.add_argument(
        "--chunk-file", "-f",
        type=str,
        default=None,
        help="Single chunk file to process"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for generated summaries"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of chunks to process"
    )
    
    args = parser.parse_args()
    
    if not args.chunks_dir and not args.chunk_file:
        parser.error("Either --chunks-dir or --chunk-file must be specified")
    
    if args.chunk_file:
        # Process single chunk - extract section name from path
        chunk_path = Path(args.chunk_file)
        section_name = chunk_path.parent.name
        
        result = await process_chunk(
            chunk_file=args.chunk_file,
            output_dir=Path(args.output_dir),
            section_name=section_name
        )
        
        if result['success']:
            print(f"\n✓ Summary generated!")
            print(f"  File: {result['summary_file']}")
        else:
            print(f"\n✗ Failed to generate summary: {result.get('error', 'Unknown')}")
    
    else:
        # Process directory
        await process_chunks_directory(
            chunks_dir=args.chunks_dir,
            output_dir=args.output_dir,
            limit=args.limit
        )


if __name__ == "__main__":
    asyncio.run(main())

