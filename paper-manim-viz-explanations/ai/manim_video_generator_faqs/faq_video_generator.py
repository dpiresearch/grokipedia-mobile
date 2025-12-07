#!/usr/bin/env python3
"""
FAQ Video Generator

This script takes a FAQ markdown file (exported from Cursor conversations)
and generates a single Manim explainer video with all Q&A pairs in sequence.

Each FAQ appears as: Question → Answer/Explanation → Next Question → ...

Uses NVIDIA Nemotron API with Qwen Coder 3 480B model.
"""

import argparse
import asyncio
import datetime
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "qwen/qwen3-coder-480b-a35b-instruct"

# System prompt for generating Manim code for ALL FAQs in one video
SYSTEM_PROMPT = """You are an expert at creating educational animations using Manim (Mathematical Animation Engine).
Your task is to create a SINGLE Manim scene that presents ALL FAQs in sequence as one continuous video.

CRITICAL: You MUST use the DETAILED explanations provided in the FAQ answers. Do NOT oversimplify.
Each FAQ answer contains rich content including code, math formulas, diagrams - VISUALIZE THEM ALL.

CRITICAL LAYOUT RULES:
1. Create a single class called `FAQScene` that inherits from `Scene`
2. ALWAYS position content at the TOP of screen using .to_edge(UP) for titles
3. ALWAYS use self.clear() between FAQs to completely clear the screen
4. After self.clear(), wait briefly with self.wait(0.3) before showing new content

FOR EACH FAQ, create a MULTI-PART explanation (1-2 minutes per FAQ):

```python
# === FAQ #N ===
self.clear()
self.wait(0.3)

# PART 1: Show question
faq_header = Text("FAQ #N", font_size=36, color=YELLOW).to_edge(UP)
question = Text("Question text", font_size=24, color=WHITE).next_to(faq_header, DOWN, buff=0.3)
self.play(Write(faq_header), FadeIn(question))
self.wait(1.5)
self.play(FadeOut(question))

# PART 2: Show key concept/overview (from the answer)
concept_title = Text("Key Concept", font_size=28, color=GREEN).next_to(faq_header, DOWN, buff=0.3)
concept = Text("Main idea from the answer...", font_size=20).next_to(concept_title, DOWN, buff=0.2)
self.play(FadeIn(concept_title), FadeIn(concept))
self.wait(2)
self.play(FadeOut(concept_title), FadeOut(concept))

# PART 3: Show code snippet (if answer has code)
code_title = Text("Implementation", font_size=28, color=BLUE).next_to(faq_header, DOWN, buff=0.3)
# Use Code() mobject or Text with monospace for code
code_text = Text("def forward(...):", font_size=16, font="Monospace").next_to(code_title, DOWN)
self.play(FadeIn(code_title), FadeIn(code_text))
self.wait(3)
self.play(FadeOut(code_title), FadeOut(code_text))

# PART 4: Show diagram/flowchart (if answer has one)
# Create boxes, arrows, etc. to represent the diagram

# PART 5: Show mathematical formula (if answer has math)
formula_title = Text("The Math", font_size=28, color=PURPLE).next_to(faq_header, DOWN, buff=0.3)
# Use MathTex for formulas - keep them simple
formula = MathTex(r"angle = m \\times \\theta_i", font_size=28).next_to(formula_title, DOWN)
self.play(FadeIn(formula_title), Write(formula))
self.wait(2)

# PART 6: Summary/Key takeaway
self.clear()
```

CONTENT REQUIREMENTS:
- Extract ALL key points from the FAQ answer, not just headlines
- If answer has CODE: Show the actual code snippets using Text with monospace font
- If answer has MATH: Show the formulas step-by-step with MathTex
- If answer has DIAGRAMS: Recreate them using Manim shapes (Rectangle, Arrow, etc.)
- If answer has FLOW/STEPS: Animate each step sequentially
- Include explanatory text for each visual element

MANIM BEST PRACTICES:
- Use Text() for prose, Text(font="Monospace") for code
- Use MathTex() for math - keep formulas simple (no \\mathds, \\mathbb, \\texttt)
- Use VGroup() to group related elements
- Use Rectangle, Arrow, Line for diagrams
- ALWAYS explicitly position with .to_edge(), .next_to(), .move_to()
- Use colors: BLUE (code), GREEN (concepts), YELLOW (titles), RED (warnings), PURPLE (math)
- Font sizes: 36 for headers, 24-28 for subtitles, 18-22 for content, 14-16 for code

Output ONLY the Python code, wrapped in ```python and ``` tags."""


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


def create_faq_title(question: str, max_length: int = 60) -> str:
    """Create a short title from the question."""
    # Remove markdown formatting
    title = re.sub(r'@\S+', '', question)  # Remove @mentions
    title = re.sub(r'`[^`]+`', '', title)  # Remove code blocks
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Truncate if needed
    if len(title) > max_length:
        title = title[:max_length-3] + "..."
    
    return title if title else "FAQ"


def create_faq_summary(faq: Dict, max_answer_length: int = 6000) -> str:
    """Create a detailed summary of a FAQ for the prompt, including full answer content."""
    question = faq['question']
    answer = faq['answer']
    
    # Include more of the answer - this is crucial for detailed animations
    if len(answer) > max_answer_length:
        answer = answer[:max_answer_length] + "\n[... truncated ...]"
    
    title = create_faq_title(question)
    
    return f"""
### FAQ #{faq['index']}: {title}

**Full Question:** 
{question}

**Detailed Answer (MUST be fully visualized):**
{answer}
"""


def create_combined_manim_prompt(faqs: List[Dict]) -> str:
    """Create the prompt for generating Manim code for ALL FAQs."""
    
    # Create summaries for each FAQ
    faq_summaries = []
    for faq in faqs:
        faq_summaries.append(create_faq_summary(faq))
    
    combined_faqs = "\n\n".join(faq_summaries)
    
    return f"""Create a SINGLE Manim animation presenting ALL {len(faqs)} FAQs about DeepSeek MoE transformers.

CRITICAL: Each FAQ answer below contains DETAILED explanations with:
- Code snippets (show them!)
- Mathematical formulas (animate them!)
- Diagrams/flowcharts (recreate them with shapes!)
- Step-by-step explanations (animate each step!)

YOU MUST visualize ALL this content, not just summarize it.

FOR EACH FAQ, create a MULTI-PART animated explanation:

```python
# === FAQ #N ===
self.clear()
self.wait(0.3)

# PART A: Question (5 seconds)
faq_header = Text("FAQ #N", font_size=36, color=YELLOW).to_edge(UP)
question = Text("Question text...", font_size=22).next_to(faq_header, DOWN, buff=0.3)
self.play(Write(faq_header), FadeIn(question))
self.wait(2)
self.play(FadeOut(question))

# PART B: Core Concept from answer (10+ seconds)
# Extract the main explanation and show it step by step
concept = Text("The core idea is...", font_size=20).next_to(faq_header, DOWN, buff=0.3)
self.play(FadeIn(concept))
self.wait(2)
# Add more detail...
detail = Text("This works because...", font_size=18).next_to(concept, DOWN)
self.play(FadeIn(detail))
self.wait(2)
self.play(FadeOut(concept), FadeOut(detail))

# PART C: Code visualization (if answer has code - 15+ seconds)
code_title = Text("Code Implementation", font_size=24, color=BLUE).next_to(faq_header, DOWN, buff=0.3)
# Show actual code from the answer:
code_line1 = Text("def forward(self, hidden_states):", font_size=14, font="Monospace")
code_line2 = Text("    outputs = self.model(...)", font_size=14, font="Monospace")
code_group = VGroup(code_line1, code_line2).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
code_group.next_to(code_title, DOWN, buff=0.3)
self.play(FadeIn(code_title))
self.play(FadeIn(code_line1))
self.wait(1)
self.play(FadeIn(code_line2))
self.wait(2)
self.play(FadeOut(code_title), FadeOut(code_group))

# PART D: Diagram/Flow (if answer has diagram - 15+ seconds)
# Recreate ASCII diagrams using Rectangle, Arrow, Text:
box1 = Rectangle(width=3, height=0.8, color=BLUE).shift(UP*1)
label1 = Text("Input", font_size=16).move_to(box1)
arrow = Arrow(box1.get_bottom(), box1.get_bottom() + DOWN*1.5)
box2 = Rectangle(width=3, height=0.8, color=GREEN).next_to(arrow, DOWN, buff=0)
label2 = Text("Output", font_size=16).move_to(box2)
self.play(Create(box1), Write(label1))
self.play(Create(arrow))
self.play(Create(box2), Write(label2))
self.wait(2)
self.play(FadeOut(VGroup(box1, label1, arrow, box2, label2)))

# PART E: Math formula (if answer has math - 10+ seconds)
math_title = Text("The Mathematics", font_size=24, color=PURPLE).next_to(faq_header, DOWN, buff=0.3)
# Use MathTex for formulas - show step by step
formula1 = MathTex(r"angle = m \\times \\theta_i", font_size=24).next_to(math_title, DOWN)
self.play(FadeIn(math_title), Write(formula1))
self.wait(2)
# Explain each term
explanation = Text("where m=position, theta=frequency", font_size=18).next_to(formula1, DOWN)
self.play(FadeIn(explanation))
self.wait(2)
self.play(FadeOut(math_title), FadeOut(formula1), FadeOut(explanation))

# PART F: Key takeaway
takeaway = Text("Key Takeaway: ...", font_size=22, color=GREEN).next_to(faq_header, DOWN, buff=0.5)
self.play(FadeIn(takeaway))
self.wait(2)
self.clear()
```

VIDEO STRUCTURE:
1. Title: "DeepSeek MoE - FAQ Deep Dive" (3 seconds)
2. Each FAQ: Multi-part explanation (60-90 seconds each)
3. End: "Thank You" screen

---
{combined_faqs}
---

REQUIREMENTS:
1. self.clear() between EVERY FAQ
2. ALL elements positioned with .to_edge(UP), .next_to(), .move_to()
3. Include ACTUAL content from each answer (code snippets, formulas, diagrams)
4. Each FAQ should be 60-90 seconds with multiple animated parts
5. Use Text(font="Monospace") for code, MathTex for math
6. Recreate diagrams using Rectangle, Arrow, Line, Text
7. AVOID complex LaTeX (no \\mathds, \\mathbb, \\texttt, \\mathrm)

Generate complete Manim code that thoroughly explains each FAQ."""


async def generate_manim_code(prompt: str) -> str:
    """Generate Manim code using NVIDIA Nemotron API with Qwen Coder."""
    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY
    )
    
    print("  Generating Manim code with Qwen Coder...")
    print("  (This may take a while for detailed FAQ video...)")
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,  # Slightly lower for more consistent code
        top_p=0.9,
        max_tokens=32000,  # Much larger for detailed content
    )
    
    return response.choices[0].message.content


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    # Remove thinking tags if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Try to find Python code block - handle various formats
    # Pattern 1: ```python ... ```
    pattern1 = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern1, text, flags=re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Pattern 2: ``` ... ``` (no language specifier)
    pattern2 = r"```\s*\n(.*?)```"
    matches = re.findall(pattern2, text, flags=re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Pattern 3: Code starts with ```python but no closing - strip markers
    if text.strip().startswith('```python'):
        code = text.strip()
        code = code[len('```python'):].strip()
        if code.endswith('```'):
            code = code[:-3].strip()
        return code
    
    if text.strip().startswith('```'):
        code = text.strip()
        code = code[3:].strip()  # Remove opening ```
        if code.endswith('```'):
            code = code[:-3].strip()
        return code
    
    # If no code blocks found, return the raw text
    return text.strip()


def save_manim_code(code: str, output_dir: Path) -> str:
    """Save Manim code to a Python file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / "faq_combined_scene.py"
    
    # Extract clean Python code
    clean_code = extract_python_code(code)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(clean_code)
    
    print(f"  Manim code saved to: {filepath}")
    return str(filepath)


def run_manim_scene(scene_filepath: str, output_dir: Path, quality: str = "l") -> Dict:
    """Run manim to generate video from the scene file."""
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the file to find scene class name
    with open(scene_filepath, 'r') as f:
        content = f.read()
    
    # Look for class definitions that inherit from Scene
    scene_match = re.search(r'class\s+(\w+)\s*\(\s*(?:Scene|ThreeDScene|MovingCameraScene)\s*\)', content)
    if not scene_match:
        raise ValueError("No Scene class found in the generated code")
    scene_name = scene_match.group(1)
    
    print(f"  Running manim for scene: {scene_name}")
    
    # Quality settings
    quality_flags = {
        'l': '-pql',   # 480p15
        'm': '-pqm',   # 720p30
        'h': '-pqh',   # 1080p60
    }
    quality_dirs = {
        'l': '480p15',
        'm': '720p30',
        'h': '1080p60',
    }
    
    # Run manim command
    cmd = [
        "python", "-m", "manim",
        quality_flags.get(quality, '-pql'),
        "--media_dir", str(media_dir),
        scene_filepath,
        scene_name
    ]
    
    print(f"  Running command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    
    if result.stdout:
        print(result.stdout)
    
    # Find the generated video file
    scene_file_basename = os.path.splitext(os.path.basename(scene_filepath))[0]
    
    # Look for video in manim's output structure
    source_video = None
    for qd in quality_dirs.values():
        potential_path = media_dir / "videos" / scene_file_basename / qd / f"{scene_name}.mp4"
        if potential_path.exists():
            source_video = potential_path
            break
    
    if not source_video:
        raise FileNotFoundError(f"Video not found in {media_dir}")
    
    # Copy video to output directory
    dest_video = output_dir / "faq_combined_video.mp4"
    shutil.copy2(source_video, dest_video)
    print(f"  Video copied to: {dest_video}")
    
    # Copy text assets if they exist
    texts_dir = media_dir / "texts"
    if texts_dir.exists():
        dest_texts = output_dir / "texts"
        if dest_texts.exists():
            shutil.rmtree(dest_texts)
        shutil.copytree(texts_dir, dest_texts)
    
    # Copy image assets if they exist
    images_dir = media_dir / "images"
    if images_dir.exists():
        dest_images = output_dir / "images"
        if dest_images.exists():
            shutil.rmtree(dest_images)
        shutil.copytree(images_dir, dest_images)
    
    return {
        'video': str(dest_video),
        'texts': str(output_dir / "texts") if texts_dir.exists() else None,
        'images': str(output_dir / "images") if images_dir.exists() else None
    }


async def generate_combined_faq_video(
    faq_file: str,
    output_dir: str,
    max_retries: int = 3,
    quality: str = "l",
    limit: int = None
) -> Dict:
    """
    Generate a single video containing all FAQs in sequence.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("FAQ COMBINED VIDEO GENERATOR")
    print(f"{'='*70}")
    print(f"\nFAQ file: {faq_file}")
    print(f"Output directory: {output_dir}")
    print(f"Max retries: {max_retries}")
    print(f"Quality: {quality}")
    
    # Parse FAQs
    faqs = parse_faq_file(faq_file)
    
    if limit:
        faqs = faqs[:limit]
    
    print(f"\nFound {len(faqs)} FAQ(s) to include in video:")
    for faq in faqs:
        title = create_faq_title(faq['question'], max_length=50)
        print(f"  {faq['index']}. {title}")
    
    # Save FAQ content for reference
    faq_content_file = output_path / "faq_content.md"
    with open(faq_content_file, 'w') as f:
        f.write("# DeepSeek MoE - Frequently Asked Questions\n\n")
        for faq in faqs:
            f.write(f"## FAQ #{faq['index']}: {create_faq_title(faq['question'])}\n\n")
            f.write(f"**Question:** {faq['question']}\n\n")
            f.write(f"**Answer:**\n{faq['answer']}\n\n")
            f.write("---\n\n")
    print(f"\nFAQ content saved to: {faq_content_file}")
    
    # Generate the combined video
    print(f"\n{'='*60}")
    print("Generating Combined FAQ Video")
    print(f"{'='*60}")
    
    last_error = None
    scene_filepath = None
    video_result = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n  Attempt {attempt}/{max_retries}")
            
            # Generate Manim code for all FAQs
            prompt = create_combined_manim_prompt(faqs)
            manim_code = await generate_manim_code(prompt)
            
            # Save the code
            scene_filepath = save_manim_code(manim_code, output_path)
            
            # Run manim to generate video
            video_result = run_manim_scene(scene_filepath, output_path, quality)
            
            # Success!
            print(f"\n  ✓ SUCCESS! Combined video generated on attempt {attempt}/{max_retries}")
            
            return {
                'faq_count': len(faqs),
                'output_dir': str(output_path),
                'scene_file': scene_filepath,
                'video_path': video_result['video'],
                'faq_content_file': str(faq_content_file),
                'success': True,
                'attempts': attempt
            }
            
        except subprocess.CalledProcessError as e:
            last_error = e
            print(f"\n  ✗ FAILED: Manim execution error")
            print(f"    Error: {e.stderr[:500] if e.stderr else str(e)}")
            
            if attempt < max_retries:
                print(f"    Retrying... ({max_retries - attempt} attempts remaining)")
            
        except Exception as e:
            last_error = e
            print(f"\n  ✗ FAILED: {type(e).__name__}")
            print(f"    Error: {str(e)[:500]}")
            
            if attempt < max_retries:
                print(f"    Retrying... ({max_retries - attempt} attempts remaining)")
    
    # All retries exhausted
    print(f"\n  ✗ Max retries ({max_retries}) reached. Video generation failed.")
    
    return {
        'faq_count': len(faqs),
        'output_dir': str(output_path),
        'scene_file': scene_filepath,
        'video_path': None,
        'success': False,
        'attempts': max_retries,
        'error': str(last_error)
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Generate a single Manim video with all FAQs in sequence"
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
        help="Output directory for generated video"
    )
    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)"
    )
    parser.add_argument(
        "--quality", "-q",
        type=str,
        choices=['l', 'm', 'h'],
        default='l',
        help="Video quality: l=480p, m=720p, h=1080p (default: l)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of FAQs to include"
    )
    
    args = parser.parse_args()
    
    result = await generate_combined_faq_video(
        faq_file=args.faq_file,
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        quality=args.quality,
        limit=args.limit
    )
    
    # Print final summary
    print(f"\n\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    
    if result['success']:
        print(f"\n✓ Successfully generated combined FAQ video!")
        print(f"  FAQs included: {result['faq_count']}")
        print(f"  Video: {result['video_path']}")
        print(f"  FAQ content: {result['faq_content_file']}")
        print(f"  Manim code: {result['scene_file']}")
    else:
        print(f"\n✗ Failed to generate video")
        print(f"  Error: {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
