import { NextRequest, NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join, resolve } from 'path'
import { existsSync } from 'fs'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ chunkDir: string }> }
) {
  try {
    const { chunkDir } = await params
    const PROJECT_ROOT = resolve(process.cwd(), '..')
    const summaryPath = join(
      PROJECT_ROOT,
      'paper-manim-viz-explanations/deepseek-moe-explainer/chunk-summary/spmoe_architecture',
      `${chunkDir}_summary.md`
    )
    
    if (!existsSync(summaryPath)) {
      return NextResponse.json({ error: 'Content not found' }, { status: 404 })
    }
    
    const content = await readFile(summaryPath, 'utf-8')
    
    return NextResponse.json({ content })
  } catch (error) {
    console.error('Error reading content:', error)
    return NextResponse.json({ error: 'Failed to read content' }, { status: 500 })
  }
}

