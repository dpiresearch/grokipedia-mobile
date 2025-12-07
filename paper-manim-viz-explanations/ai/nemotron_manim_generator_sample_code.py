import asyncio
import os
import sys
from pathlib import Path
import requests
import re
import datetime
import subprocess
import argparse

# Add the backend directory to the path so we can import from db
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import httpx
from dotenv import load_dotenv
import fal_client
from db.db import store_media, create_article, get_article_by_id
from ai.scrape import get_article
from utils.s3_upload import upload_to_s3

load_dotenv()

async def generate_prompt(prompt:str="", system_prompt:str=""):
    """Generate a prompt using the Qwen model based on an article content or URL"""
    # Create a custom http client without proxies to avoid compatibility issues
    http_client = httpx.Client(base_url="https://integrate.api.nvidia.com/v1")
    
    # Initialize the OpenAI client with the custom http client
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = os.getenv("NVIDIA_API_KEY"),
        http_client = http_client
    )

    completion = client.chat.completions.create(
        model="qwen/qwen3-next-80b-a3b-thinking",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content": prompt}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=True
    )

    full_content = ""
    reasoning_done = False
    for chunk in completion:
        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="")
            reasoning_done = True
        elif reasoning_done and chunk.choices[0].delta.content is not None:
            print("\n--- END OF REASONING ---\n", end="")
            reasoning_done = False
        
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="")
            full_content += content
    
    return full_content

def extract_concepts(text: str) -> list[str]:
    pattern = r"<concept>\s*(.*?)\s*</concept>"
    return re.findall(pattern, text, flags=re.DOTALL)

async def decompose_article(article):
    "decompose article to concepts"
    prompt = f"""decompose this article into a series of important concepts. Keep it concise but factual
    {article}
    Return the concepts in a series of <concept> </concept> tags

    Example:
    <concept> Here is a concept </concept>
    """
    res = await generate_prompt(prompt=prompt)
    concepts = extract_concepts(res)
    print(res, concepts)
    return concepts

async def generate_manim_code(prompt:str="", system_prompt:str=""):
    """Generate manim code using the Qwen coder model"""
    # Create a custom http cl
    # ient without proxies to avoid compatibility issues
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = os.getenv("NVIDIA_API_KEY")
    )

    completion = client.chat.completions.create(
        model="qwen/qwen3-coder-480b-a35b-instruct",
        messages=[
            {"role":"system","content": system_prompt},
            {"role":"user","content": prompt}
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=4096,
        stream=True
    )

    full_content = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            full_content += chunk.choices[0].delta.content
    
    return full_content

def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks or raw text"""
    # Try to extract from markdown code blocks first
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[0].strip()
    # If no code blocks found, return the raw text
    return text.strip()

def save_manim_code(code: str, output_dir: str = "manim/code") -> str:
    """Save manim code to a Python file"""
    # Create output directory if it doesn't exist (relative to backend directory)
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scene_{timestamp}.py"
    filepath = output_path / filename
    
    # Extract clean Python code
    clean_code = extract_python_code(code)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(clean_code)
    
    print(f"\n=== Manim code saved to: {filepath} ===")
    return str(filepath)

async def run_manim_scene(scene_filepath: str, output_dir: str = "manim/generated_video") -> str:
    """Run manim to generate video from the scene file"""
    # Create output directory (relative to backend directory)
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract scene class name from the file
    with open(scene_filepath, 'r') as f:
        content = f.read()
        # Look for class definitions that inherit from Scene
        scene_match = re.search(r'class\s+(\w+)\s*\(\s*Scene\s*\)', content)
        if not scene_match:
            raise ValueError("No Scene class found in the generated code")
        scene_name = scene_match.group(1)
    
    print(f"\n=== Running manim for scene: {scene_name} ===")
    
    # Run manim command
    # -pql = preview, quality low (for faster generation)
    # Can change to -pqh for high quality
    cmd = [
        "manim",
        "-pql",  # preview, quality low
        "--media_dir", str(output_path),
        scene_filepath,
        scene_name
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        
        # Find the generated video file
        # Manim outputs to media_dir/videos/scene_filename/quality/SceneName.mp4
        scene_file_basename = os.path.splitext(os.path.basename(scene_filepath))[0]
        video_path = output_path / "videos" / scene_file_basename / "480p15" / f"{scene_name}.mp4"
        
        if video_path.exists():
            print(f"\n=== Video generated at: {video_path} ===")
            return str(video_path)
        else:
            raise FileNotFoundError(f"Expected video not found at {video_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running manim: {e.stderr}")
        raise

async def create_generation_prompt(concept, max_length, style="manim"):
    # Create manim code generation prompt
    prompt = f"""Generate complete, executable Manim Python code to create an educational animation explaining this concept:

'{concept[:max_length]}'

Requirements:
- Create a Scene class that inherits from Scene
- Use Text() for regular text (NOT Tex()). Text() is faster and doesn't require LaTeX
- ONLY use MathTex() or Tex() for mathematical formulas and equations
- Use appropriate Manim shapes (Circle, Rectangle, Square, Arrow, Line, Dot, etc.)
- Include smooth animations (Write, FadeIn, FadeOut, Transform, Create, GrowArrow, etc.)
- Break down the concept into 3-5 clear visual steps
- Use colors effectively (BLUE, RED, GREEN, YELLOW, etc.)
- Position elements using .to_edge(), .shift(), .next_to(), .move_to()
- Include self.wait() between major steps (0.5-1 second)
- CRITICAL: Use FadeOut() or self.remove() to clean up previous content before showing new content
- Group related objects in VGroup() so they can be animated together
- Between major steps, fade out old content with self.play(FadeOut(old_objects))
- Keep the scene clean - don't let objects accumulate and clutter the frame
- Make it educational and engaging with clear visual metaphors
- Ensure the code is syntactically correct and ready to run
- The animation should be 10-30 seconds long
- Import only what you need: from manim import *

Return ONLY the Python code, no explanations."""
    
    system_prompt = """You are an expert in creating educational animations using Manim (Mathematical Animation Engine), similar to 3Blue1Brown style. You specialize in:
- Breaking down complex concepts into visual steps with clear narratives
- Using Text() for all regular text (faster, no LaTeX compilation needed)
- Using MathTex() ONLY for mathematical formulas (e.g., equations, symbols)
- Using appropriate Manim primitives (Circle, Rectangle, Arrow, VGroup, NumberLine, Axes)
- Creating smooth, educational animations with proper timing and wait periods
- CLEANING UP content between steps using FadeOut() to prevent visual clutter
- Grouping related objects in VGroup() for easier animation and removal
- Writing clean, executable Manim code following best practices
- Using the Scene class with proper construct() methods
- Implementing effective visual metaphors and transitions
- Using colors to highlight important concepts
- Following Manim Community Edition (manim-ce) syntax

Generate production-ready code that can be directly executed with: manim -pql scene.py SceneName

CRITICAL RULES:
1. Use Text() for words and sentences. Use MathTex() ONLY for mathematical notation.
2. Always fade out or remove previous content before introducing new content in each step.
3. Keep the scene clean - use self.play(FadeOut(objects)) between major transitions.
4. AVOID Code() objects - they have complex parameters that often fail. Use Text() with monospace styling instead.
5. Test your knowledge - only use Manim Community Edition v0.19.0 compatible methods and parameters."""
    
    code = await generate_manim_code(prompt, system_prompt)
    return code

async def generate_image(prompt):
    """Generate an image using fal-ai API"""
    if not prompt:
        print("\nError: Empty prompt")
        return None

    print(f"\n\nGenerating image with prompt: {prompt}\n")
    
    handler = await fal_client.submit_async(
        "fal-ai/alpha-image-232/text-to-image",
        arguments={
            "prompt": prompt
        },
    )
    
    async for event in handler.iter_events(with_logs=True):
        print(event)
    
    result = await handler.get()
    return result

async def process_article_and_generate_media(article_url=None, style="manim", user_id=1, max_retries=5):
    """Process an article and generate manim video content, storing results in the database
    
    Args:
        article_url: URL of the article to process
        style: Generation style (default: "manim")
        user_id: User ID for database storage
        max_retries: Maximum number of code generation attempts (default: 5)
    """
    
    # Get article and extract concepts
    article_text = get_article(article_url)
    concepts = await decompose_article(article_text)
    concept = '\n'.join([f'\n Concept {i+1}: {concept}\n' for i, concept in enumerate(concepts)])
    
    for attempt in range(1, max_retries + 1):
        try:
            # Generate manim code
            manim_code = await create_generation_prompt(concept=concept, max_length=500)
            
            # Save the code to a file
            scene_filepath = save_manim_code(manim_code)
            
            # Run manim to generate video
            video_path = await run_manim_scene(scene_filepath)
            
            # If we got here, video was generated successfully
            print(f"\n{'='*60}")
            print(f"SUCCESS! Video generated on attempt {attempt}/{max_retries}")
            print(f"{'='*60}")
            break
            
        except subprocess.CalledProcessError as e:
            last_error = e
            print(f"\n{'='*60}")
            print(f"FAILED: Attempt {attempt}/{max_retries} - Manim execution error")
            print(f"Error: {e}")
            print(f"{'='*60}")
            
            if attempt < max_retries:
                print(f"\nRetrying... ({max_retries - attempt} attempts remaining)")
            else:
                print(f"\nMax retries ({max_retries}) reached. Giving up.")
                raise RuntimeError(
                    f"Failed to generate valid Manim code after {max_retries} attempts. "
                    f"Last error: {e}"
                ) from e
                
        except Exception as e:
            last_error = e
            print(f"\n{'='*60}")
            print(f"FAILED: Attempt {attempt}/{max_retries} - Unexpected error")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {e}")
            print(f"{'='*60}")
            
            if attempt < max_retries:
                print(f"\nRetrying... ({max_retries - attempt} attempts remaining)")
            else:
                print(f"\nMax retries ({max_retries}) reached. Giving up.")
                raise RuntimeError(
                    f"Failed to generate valid Manim code after {max_retries} attempts. "
                    f"Last error: {e}"
                ) from e
    
    # Create article in database
    article = await create_article(article_url, text="\n".join(concepts), user_id=user_id)
    article_id = article["id"]
    
    # Upload video to S3
    print("\n=== Uploading video to S3 ===")
    try:
        s3_url = upload_to_s3(video_path, s3_folder="manim_videos")
        print(f"Video uploaded to S3: {s3_url}")
        media_url = s3_url
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        print("Falling back to local path")
        media_url = video_path
    
    # Store the media in the database
    media_row = await store_media(
        article_id=article_id,
        prompt=concept[:500],  # Store the concept as the prompt
        style=style,
        media_type="video",
        media_url=media_url  # S3 URL or local path as fallback
    )
    
    print(f"\n=== Media stored in database with ID {media_row['id']} ===")
    
    return {
        "article_id": article_id,
        "media_id": media_row["id"],
        "concept": concept,
        "video_path": media_url,  # Return S3 URL
        "scene_file": scene_filepath
    }
    

async def main(max_retries=5):
    """Main function to process article and generate video
    
    Args:
        max_retries: Maximum number of code generation attempts (default: 5)
    """
    # Example usage
    article_url = "https://www.lesswrong.com/posts/fJtELFKddJPfAxwKS/natural-emergent-misalignment-from-reward-hacking-in"
    result = await process_article_and_generate_media(
        article_url=article_url, 
        max_retries=max_retries
    )
    print(result)

async def test():
    article_url = "https://www.lesswrong.com/posts/fJtELFKddJPfAxwKS/natural-emergent-misalignment-from-reward-hacking-in"
    res = get_article(article_url)
    print(res)
    concepts = await decompose_article(res)
    print(concepts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Manim educational videos from articles"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of code generation attempts if generation fails (default: 5)"
    )
    args = parser.parse_args()
    
    asyncio.run(main(max_retries=args.max_retries))
