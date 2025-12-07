import { NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'
import { existsSync } from 'fs'

const PROJECT_ROOT = process.cwd()
const FAQ_CONTENT_PATH = join(
  PROJECT_ROOT,
  'paper-manim-viz-explanations/deepseek-moe-explainer/faq-summary/cursor_deepseek_model_forward_pass_in_h_comprehensive_summary.md'
)

export async function GET() {
  try {
    if (!existsSync(FAQ_CONTENT_PATH)) {
      return NextResponse.json({ error: 'FAQ content not found' }, { status: 404 })
    }
    
    const content = await readFile(FAQ_CONTENT_PATH, 'utf-8')
    
    return NextResponse.json({ content })
  } catch (error) {
    console.error('Error reading FAQ content:', error)
    return NextResponse.json({ error: 'Failed to read FAQ content' }, { status: 500 })
  }
}

