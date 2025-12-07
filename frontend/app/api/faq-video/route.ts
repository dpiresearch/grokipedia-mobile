import { NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join, resolve } from 'path'
import { existsSync } from 'fs'

const PROJECT_ROOT = resolve(process.cwd(), '..')
const FAQ_VIDEO_PATH = join(
  PROJECT_ROOT,
  'paper-manim-viz-explanations/deepseek-moe-explainer/generated_videos/faq_generated_videos_v3_full/faq_combined_video.mp4'
)

export async function GET() {
  try {
    if (!existsSync(FAQ_VIDEO_PATH)) {
      return NextResponse.json({ error: 'FAQ video not found' }, { status: 404 })
    }
    
    const videoBuffer = await readFile(FAQ_VIDEO_PATH)
    
    return new NextResponse(videoBuffer, {
      headers: {
        'Content-Type': 'video/mp4',
        'Content-Length': videoBuffer.length.toString(),
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    })
  } catch (error) {
    console.error('Error serving FAQ video:', error)
    return NextResponse.json({ error: 'Failed to serve FAQ video' }, { status: 500 })
  }
}

