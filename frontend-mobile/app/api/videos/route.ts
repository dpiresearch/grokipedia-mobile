import { NextResponse } from 'next/server'
import { readdir, readFile } from 'fs/promises'
import { join } from 'path'
import { existsSync } from 'fs'

// Resolve paths relative to the current directory (paper-manim-viz-explanations is copied during build)
const PROJECT_ROOT = process.cwd()
const VIDEOS_DIR = join(PROJECT_ROOT, 'paper-manim-viz-explanations/deepseek-moe-explainer/generated_videos/spmoe_architecture')
const SUMMARIES_DIR = join(PROJECT_ROOT, 'paper-manim-viz-explanations/deepseek-moe-explainer/chunk-summary/spmoe_architecture')

interface VideoItem {
  id: string
  title: string
  description: string
  videoPath: string
  chunkNumber: number
  header: string
}

export async function GET() {
  try {
    if (!existsSync(VIDEOS_DIR)) {
      console.error(`VIDEOS_DIR does not exist: ${VIDEOS_DIR}`)
      return NextResponse.json({ error: `Videos directory not found: ${VIDEOS_DIR}` }, { status: 500 })
    }
    if (!existsSync(SUMMARIES_DIR)) {
      console.error(`SUMMARIES_DIR does not exist: ${SUMMARIES_DIR}`)
      return NextResponse.json({ error: `Summaries directory not found: ${SUMMARIES_DIR}` }, { status: 500 })
    }
    
    const videoItems: VideoItem[] = []
    
    const chunkDirs = await readdir(VIDEOS_DIR, { withFileTypes: true })
    const chunkDirsList = chunkDirs
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name)
      .filter(name => name.startsWith('chunk_'))
      .sort()

    for (const chunkDir of chunkDirsList) {
      const match = chunkDir.match(/chunk_(\d+)_(.+)/)
      if (!match) continue
      
      const chunkNumber = parseInt(match[1], 10)
      const chunkName = match[2]
      
      const videoPath = join(VIDEOS_DIR, chunkDir, 'video.mp4')
      if (!existsSync(videoPath)) continue
      
      const summaryPath = join(SUMMARIES_DIR, `${chunkDir}_summary.md`)
      let title = chunkName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
      let description = ''
      const summaryFilename = `${chunkDir}_summary.md`
      const headerMatch = summaryFilename.match(/chunk_(.+)\.md/)
      const header = headerMatch ? headerMatch[1] : chunkName
      
      if (existsSync(summaryPath)) {
        try {
          const summaryContent = await readFile(summaryPath, 'utf-8')
          const titleMatch = summaryContent.match(/^#\s+(.+)$/m)
          if (titleMatch) {
            title = titleMatch[1]
          }
          const paragraphs = summaryContent.split('\n\n').filter(p => p.trim() && !p.startsWith('#'))
          if (paragraphs.length > 0) {
            description = paragraphs[0].replace(/\n/g, ' ').trim().substring(0, 200)
            if (paragraphs[0].length > 200) {
              description += '...'
            }
          }
        } catch (error) {
          console.error(`Error reading summary for ${chunkDir}:`, error)
        }
      }
      
      videoItems.push({
        id: `chunk_${chunkNumber}`,
        title,
        description,
        videoPath: `/api/video/${chunkDir}`,
        chunkNumber,
        header,
      })
    }
    
    return NextResponse.json({ videos: videoItems })
  } catch (error) {
    console.error('Error loading videos:', error)
    return NextResponse.json({ error: 'Failed to load videos' }, { status: 500 })
  }
}

