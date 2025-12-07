import { NextResponse } from 'next/server'
import { readdir, readFile } from 'fs/promises'
import { join } from 'path'
import { existsSync } from 'fs'

const VIDEOS_DIR = join(process.cwd(), 'paper-manim-viz-explanations/deepseek-moe-explainer/generated_videos/spmoe_architecture')
const SUMMARIES_DIR = join(process.cwd(), 'paper-manim-viz-explanations/deepseek-moe-explainer/chunk-summary/spmoe_architecture')

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
    const videoItems: VideoItem[] = []
    
    // Get all chunk directories
    const chunkDirs = await readdir(VIDEOS_DIR, { withFileTypes: true })
    const chunkDirsList = chunkDirs
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name)
      .filter(name => name.startsWith('chunk_'))
      .sort()

    for (const chunkDir of chunkDirsList) {
      // Extract chunk number and name
      const match = chunkDir.match(/chunk_(\d+)_(.+)/)
      if (!match) continue
      
      const chunkNumber = parseInt(match[1], 10)
      const chunkName = match[2]
      
      // Check if video exists
      const videoPath = join(VIDEOS_DIR, chunkDir, 'video.mp4')
      if (!existsSync(videoPath)) continue
      
      // Read corresponding markdown summary
      const summaryPath = join(SUMMARIES_DIR, `${chunkDir}_summary.md`)
      let title = chunkName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
      let description = ''
      // Extract header from filename: text between chunk_ and .md
      // For chunk_01_deepseekmoe_overview_summary.md, header is 01_deepseekmoe_overview_summary
      const summaryFilename = `${chunkDir}_summary.md`
      const headerMatch = summaryFilename.match(/chunk_(.+)\.md/)
      const header = headerMatch ? headerMatch[1] : chunkName
      
      if (existsSync(summaryPath)) {
        try {
          const summaryContent = await readFile(summaryPath, 'utf-8')
          // Extract title from first heading
          const titleMatch = summaryContent.match(/^#\s+(.+)$/m)
          if (titleMatch) {
            title = titleMatch[1]
          }
          // Extract first paragraph as description
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

