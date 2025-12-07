"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Upload, Link, FileText, Play, Loader2 } from "lucide-react"
import { InlineMath, BlockMath } from 'react-katex'
import 'katex/dist/katex.min.css'

interface VideoItem {
  id: string
  title: string
  description: string
  videoPath: string
  chunkNumber: number
  header: string
}

export function PaperSubmissionPage() {
  const [arxivUrl, setArxivUrl] = useState("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [videos, setVideos] = useState<VideoItem[]>([])
  const [faqs, setFaqs] = useState<VideoItem[]>([])
  const [isLoadingVideos, setIsLoadingVideos] = useState(false)
  const [urlError, setUrlError] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type === "application/pdf") {
      setSelectedFile(file)
      setArxivUrl("")
    }
  }

  const validateUrl = (url: string): boolean => {
    if (!url.trim()) return true // Empty is valid (not required if file is selected)
    try {
      const urlObj = new URL(url)
      // Check if it's an arxiv.org URL
      return urlObj.hostname === 'arxiv.org' || urlObj.hostname === 'www.arxiv.org'
    } catch {
      return false
    }
  }

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setArxivUrl(value)
    
    // Validate URL
    if (value.trim() && !validateUrl(value)) {
      setUrlError("Please enter a valid ArXiv URL (e.g., https://arxiv.org/abs/...)")
    } else {
      setUrlError("")
    }
    
    if (value) {
      setSelectedFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  const handleSubmit = async () => {
    if (!arxivUrl && !selectedFile) return
    
    // Validate URL if provided
    if (arxivUrl.trim() && !validateUrl(arxivUrl)) {
      setUrlError("Please enter a valid ArXiv URL (e.g., https://arxiv.org/abs/...)")
      return
    }

    setIsSubmitting(true)
    setIsLoadingVideos(true)
    setVideos([])
    setFaqs([])
    
    // Wait 2 seconds before loading content (with animation)
    await new Promise((resolve) => setTimeout(resolve, 2000))
    
    // Load videos and FAQs after delay
    try {
      // Load videos
      const videosResponse = await fetch('/api/videos')
      if (videosResponse.ok) {
        const videosData = await videosResponse.json()
        setVideos(videosData.videos || [])
      }
      
      // Load FAQ (single item)
      const faqItem: VideoItem = {
        id: 'faq_1',
        title: 'DeepSeek MoE Transformer: Forward Pass and Context Mechanics',
        description: 'Comprehensive exploration of DeepSeek\'s Mixture of Experts transformer implementation, focusing on forward pass architecture, context handling mechanisms, and mathematical foundations.',
        videoPath: '/api/faq-video',
        chunkNumber: 0,
        header: 'faq_combined_video',
      }
      setFaqs([faqItem])
    } catch (error) {
      console.error('Error loading videos:', error)
    } finally {
      setIsSubmitting(false)
      setIsLoadingVideos(false)
    }
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-7xl">
        <h1 className="mb-8 text-center text-3xl font-bold text-foreground">Paper to Video Generator</h1>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Left Pane - Input Section */}
          <Card className="h-fit">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Submit Paper
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* URL Input */}
              <div className="space-y-2">
                <Label htmlFor="arxiv-url" className="flex items-center gap-2">
                  <Link className="h-4 w-4" />
                  ArXiv URL
                </Label>
                <Input
                  id="arxiv-url"
                  type="url"
                  placeholder="https://arxiv.org/abs/..."
                  value={arxivUrl}
                  onChange={handleUrlChange}
                  disabled={isSubmitting}
                  className={urlError ? "border-destructive" : ""}
                />
                {urlError && (
                  <p className="text-sm text-destructive">{urlError}</p>
                )}
              </div>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-card px-2 text-muted-foreground">Or</span>
                </div>
              </div>

              {/* File Upload */}
              <div className="space-y-2">
                <Label htmlFor="pdf-upload" className="flex items-center gap-2">
                  <Upload className="h-4 w-4" />
                  Upload PDF
                </Label>
                <div
                  className="cursor-pointer rounded-lg border-2 border-dashed border-input p-6 text-center transition-colors hover:border-primary"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    id="pdf-upload"
                    type="file"
                    accept=".pdf"
                    className="hidden"
                    onChange={handleFileChange}
                    disabled={isSubmitting}
                  />
                  {selectedFile ? (
                    <div className="flex flex-col items-center gap-2">
                      <FileText className="h-10 w-10 text-primary" />
                      <p className="text-sm font-medium text-foreground">{selectedFile.name}</p>
                      <p className="text-xs text-muted-foreground">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-2">
                      <Upload className="h-10 w-10 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">Click to upload or drag and drop</p>
                      <p className="text-xs text-muted-foreground">PDF files only</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Submit Button */}
              <Button
                className="w-full"
                size="lg"
                onClick={handleSubmit}
                disabled={(!arxivUrl && !selectedFile) || isSubmitting}
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing Paper...
                  </>
                ) : (
                  "Submit Paper"
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Right Pane - Videos and FAQs */}
          <div className="flex flex-col gap-6">
            {/* Videos Section */}
            <Card className="flex-1">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Play className="h-5 w-5" />
                  Videos
                  {videos.length > 0 && (
                    <span className="ml-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-normal text-primary">
                      {videos.length}
                    </span>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80 overflow-y-auto pr-2">
                  {isLoadingVideos || isSubmitting ? (
                    <div className="flex h-full items-center justify-center text-muted-foreground">
                      <div className="flex flex-col items-center gap-4">
                        <div className="relative">
                          <Loader2 className="h-12 w-12 animate-spin text-primary" />
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="h-8 w-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                          </div>
                        </div>
                        <div className="text-center space-y-1">
                          <p className="text-sm font-medium">Processing your paper...</p>
                          <p className="text-xs text-muted-foreground">Generating videos and content</p>
                        </div>
                        <div className="flex gap-1">
                          <div className="h-2 w-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                          <div className="h-2 w-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                          <div className="h-2 w-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                      </div>
                    </div>
                  ) : videos.length === 0 ? (
                    <div className="flex h-full items-center justify-center text-muted-foreground">
                      <p className="text-center text-sm">No videos available</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {videos.map((video) => (
                        <VideoCard key={video.id} video={video} />
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* FAQs Section */}
            <Card className="flex-1">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  FAQs
                  {faqs.length > 0 && (
                    <span className="ml-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-normal text-primary">
                      {faqs.length}
                    </span>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80 overflow-y-auto pr-2">
                  {isLoadingVideos || isSubmitting ? (
                    <div className="flex h-full items-center justify-center text-muted-foreground">
                      <div className="flex flex-col items-center gap-4">
                        <div className="relative">
                          <Loader2 className="h-12 w-12 animate-spin text-primary" />
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="h-8 w-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                          </div>
                        </div>
                        <div className="text-center space-y-1">
                          <p className="text-sm font-medium">Processing your paper...</p>
                          <p className="text-xs text-muted-foreground">Generating FAQ content</p>
                        </div>
                        <div className="flex gap-1">
                          <div className="h-2 w-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                          <div className="h-2 w-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                          <div className="h-2 w-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                      </div>
                    </div>
                  ) : faqs.length === 0 ? (
                    <div className="flex h-full items-center justify-center text-muted-foreground">
                      <p className="text-center text-sm">No FAQ videos available</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {faqs.map((faq) => (
                        <VideoCard key={faq.id} video={faq} />
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

function VideoCard({ video }: { video: VideoItem }) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [fullContent, setFullContent] = useState<string>("")
  const [isLoadingContent, setIsLoadingContent] = useState(false)
  
  // Format header: replace underscores with spaces and capitalize words
  const formattedHeader = video.header
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (l) => l.toUpperCase())
  
  // Determine if this is an FAQ video or regular video
  const isFaqVideo = video.videoPath === '/api/faq-video'
  
  const handleWatchClick = async () => {
    setIsModalOpen(true)
    setIsLoadingContent(true)
    
    try {
      // Use different API endpoint for FAQ vs regular videos
      const apiEndpoint = isFaqVideo ? '/api/faq-content' : `/api/video-content/${video.videoPath.replace('/api/video/', '')}`
      const response = await fetch(apiEndpoint)
      if (response.ok) {
        const data = await response.json()
        setFullContent(data.content || "")
      } else {
        setFullContent("Content not available.")
      }
    } catch (error) {
      console.error('Error loading content:', error)
      setFullContent("Error loading content.")
    } finally {
      setIsLoadingContent(false)
    }
  }
  
  return (
    <>
      <div className="flex flex-col gap-3 rounded-lg border bg-card p-3 transition-colors hover:bg-muted/50">
        <div className="flex gap-4">
          <div className="relative flex-shrink-0 w-40 h-24 bg-muted rounded-md overflow-hidden">
            {isPlaying ? (
              <video
                src={video.videoPath}
                controls
                className="w-full h-full object-cover"
                onPause={() => setIsPlaying(false)}
              />
            ) : (
              <>
                <div className="absolute inset-0 flex items-center justify-center bg-muted">
                  <Play className="h-8 w-8 text-muted-foreground" />
                </div>
                <div
                  className="absolute inset-0 cursor-pointer"
                  onClick={() => setIsPlaying(true)}
                />
              </>
            )}
          </div>
          <div className="flex flex-col justify-center overflow-hidden flex-1">
            <h4 className="text-xs font-medium text-muted-foreground mb-1 uppercase tracking-wide">
              {formattedHeader}
            </h4>
            <h3 className="font-semibold text-foreground mb-1">{video.title}</h3>
            <p className="line-clamp-3 text-sm text-muted-foreground mb-2">{video.description}</p>
            <Button
              variant="outline"
              size="sm"
              onClick={handleWatchClick}
              className="w-fit"
            >
              <Play className="h-3 w-3 mr-1" />
              Watch
            </Button>
          </div>
        </div>
      </div>
      
      <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
        <DialogContent className="max-w-5xl max-h-[90vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>{video.title}</DialogTitle>
            <DialogDescription className="text-xs uppercase tracking-wide">
              {formattedHeader}
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-4 flex-1 min-h-0">
            {/* Video Section */}
            <div className="w-full h-96 bg-muted rounded-lg overflow-hidden flex-shrink-0">
              <video
                src={video.videoPath}
                controls
                className="w-full h-full object-contain"
                autoPlay
              />
            </div>
            
            {/* Text Section - Scrollable */}
            <div className="flex-1 min-h-0 flex flex-col">
              <h3 className="text-lg font-semibold mb-2">Content</h3>
              <div className="flex-1 overflow-y-auto border rounded-lg p-4 bg-muted/30">
                {isLoadingContent ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  </div>
                ) : (
                  <MarkdownContent content={fullContent} />
                )}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}

function MarkdownContent({ content }: { content: string }) {
  const renderLine = (line: string, index: number) => {
    // Handle code blocks
    if (line.startsWith('```')) {
      return <br key={index} />
    }
    
    // Handle headings
    if (line.startsWith('# ')) {
      return <h1 key={index} className="text-2xl font-bold mt-4 mb-2">{renderInlineMath(line.substring(2))}</h1>
    } else if (line.startsWith('## ')) {
      return <h2 key={index} className="text-xl font-bold mt-3 mb-2">{renderInlineMath(line.substring(3))}</h2>
    } else if (line.startsWith('### ')) {
      return <h3 key={index} className="text-lg font-semibold mt-2 mb-1">{renderInlineMath(line.substring(4))}</h3>
    } else if (line.trim() === '') {
      return <br key={index} />
    }
    
    // Handle block math ($$...$$)
    const blockMathRegex = /\$\$(.+?)\$\$/g
    if (blockMathRegex.test(line)) {
      const parts: (string | JSX.Element)[] = []
      let lastIndex = 0
      let match
      blockMathRegex.lastIndex = 0
      
      while ((match = blockMathRegex.exec(line)) !== null) {
        if (match.index > lastIndex) {
          parts.push(renderInlineMath(line.substring(lastIndex, match.index)))
        }
        try {
          parts.push(<BlockMath key={`block-${match.index}`} math={match[1]} />)
        } catch (e) {
          parts.push(<span key={`block-${match.index}`}>$${match[1]}$$</span>)
        }
        lastIndex = match.index + match[0].length
      }
      if (lastIndex < line.length) {
        parts.push(renderInlineMath(line.substring(lastIndex)))
      }
      return <div key={index} className="mb-2">{parts}</div>
    }
    
    // Regular paragraph with inline math and bold
    return (
      <p key={index} className="mb-2">
        {renderInlineMath(line)}
      </p>
    )
  }
  
  const renderInlineMath = (text: string) => {
    const parts: (string | JSX.Element)[] = []
    // Match inline math ($...$) but not block math ($$...$$)
    // Use a simpler approach: find $ that's not preceded or followed by another $
    let lastIndex = 0
    let dollarIndex = -1
    
    for (let i = 0; i < text.length; i++) {
      if (text[i] === '$') {
        // Check if it's part of block math ($$)
        if (i < text.length - 1 && text[i + 1] === '$') {
          i++ // Skip both $$
          continue
        }
        // Check if previous char was also $
        if (i > 0 && text[i - 1] === '$') {
          continue // Skip this $ as it's part of $$
        }
        
        if (dollarIndex === -1) {
          // Opening $
          if (i > lastIndex) {
            parts.push(...renderBold(text.substring(lastIndex, i)))
          }
          dollarIndex = i
        } else {
          // Closing $
          const mathContent = text.substring(dollarIndex + 1, i)
          try {
            parts.push(<InlineMath key={`math-${dollarIndex}`} math={mathContent} />)
          } catch (e) {
            parts.push(<span key={`math-${dollarIndex}`}>${mathContent}$</span>)
          }
          lastIndex = i + 1
          dollarIndex = -1
        }
      }
    }
    
    if (dollarIndex !== -1) {
      // Unclosed math, treat as regular text
      parts.push(...renderBold(text.substring(lastIndex)))
    } else if (lastIndex < text.length) {
      parts.push(...renderBold(text.substring(lastIndex)))
    }
    
    return parts.length > 0 ? parts : renderBold(text)
  }
  
  const renderBold = (text: string): (string | JSX.Element)[] => {
    const parts: (string | JSX.Element)[] = []
    const boldRegex = /\*\*(.+?)\*\*/g
    let lastIndex = 0
    let match
    
    while ((match = boldRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index))
      }
      parts.push(<strong key={`bold-${match.index}`}>{match[1]}</strong>)
      lastIndex = match.index + match[0].length
    }
    
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex))
    }
    
    return parts.length > 0 ? parts : [text]
  }
  
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      <div className="font-sans text-sm text-foreground leading-relaxed">
        {content.split('\n').map((line, index) => renderLine(line, index))}
      </div>
    </div>
  )
}
