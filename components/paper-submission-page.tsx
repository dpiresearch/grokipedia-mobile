"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, Link, FileText, Play, Loader2 } from "lucide-react"

interface VideoItem {
  id: string
  title: string
  description: string
  videoUrl: string
  thumbnail: string
}

const mockVideos: VideoItem[] = [
  {
    id: "v1",
    title: "Paper Overview",
    description:
      "This animation provides a high-level overview of the paper's main contributions, including the novel architecture proposed and its key innovations in the field of machine learning.",
    videoUrl: "#",
    thumbnail: "/research-paper-overview-animation.jpg",
  },
  {
    id: "v2",
    title: "Methodology Explained",
    description:
      "A detailed walkthrough of the methodology section, explaining the experimental setup, data collection process, and analysis techniques used throughout the study.",
    videoUrl: "#",
    thumbnail: "/scientific-methodology-diagram.jpg",
  },
  {
    id: "v3",
    title: "Results Visualization",
    description:
      "Visual representation of the key results and findings from the experiments, including performance comparisons with baseline methods and statistical significance.",
    videoUrl: "#",
    thumbnail: "/data-visualization-charts.png",
  },
  {
    id: "v4",
    title: "Architecture Deep Dive",
    description:
      "An in-depth explanation of the neural network architecture, layer configurations, attention mechanisms, and how data flows through the model during inference.",
    videoUrl: "#",
    thumbnail: "/neural-network-architecture.png",
  },
  {
    id: "v5",
    title: "Training Process",
    description:
      "Visualization of the training process including loss curves, optimization strategies, learning rate schedules, and convergence behavior over epochs.",
    videoUrl: "#",
    thumbnail: "/training-loss-curve-graph.jpg",
  },
  {
    id: "v6",
    title: "Comparison with SOTA",
    description:
      "Side-by-side comparison of the proposed method with state-of-the-art approaches, highlighting improvements in accuracy, speed, and resource efficiency.",
    videoUrl: "#",
    thumbnail: "/benchmark-comparison-chart.jpg",
  },
]

const mockFaqs: VideoItem[] = [
  {
    id: "f1",
    title: "What problem does this paper solve?",
    description:
      "This video explains the core problem addressed by the paper and why it matters for the research community and practical applications in industry.",
    videoUrl: "#",
    thumbnail: "/question-mark-research-concept.jpg",
  },
  {
    id: "f2",
    title: "How does the proposed method work?",
    description:
      "A simplified explanation of the proposed method, breaking down complex concepts into understandable components with visual aids and step-by-step diagrams.",
    videoUrl: "#",
    thumbnail: "/algorithm-flowchart-explanation.jpg",
  },
  {
    id: "f3",
    title: "What are the limitations?",
    description:
      "An honest discussion of the limitations of the proposed approach and potential directions for future research to address them in subsequent work.",
    videoUrl: "#",
    thumbnail: "/research-limitations-warning-icon.jpg",
  },
  {
    id: "f4",
    title: "What datasets were used?",
    description:
      "Overview of the datasets used for training and evaluation, including their size, characteristics, preprocessing steps, and how they were split for experiments.",
    videoUrl: "#",
    thumbnail: "/database-dataset-collection-icon.jpg",
  },
  {
    id: "f5",
    title: "How can I reproduce the results?",
    description:
      "Step-by-step guide on reproducing the paper's results, including code availability, hardware requirements, hyperparameter settings, and expected runtime.",
    videoUrl: "#",
    thumbnail: "/code-reproducibility-checklist.jpg",
  },
  {
    id: "f6",
    title: "What are the real-world applications?",
    description:
      "Exploration of practical applications where this research can be applied, from healthcare diagnostics to autonomous systems and natural language processing.",
    videoUrl: "#",
    thumbnail: "/real-world-application-icons.jpg",
  },
]

export function PaperSubmissionPage() {
  const [arxivUrl, setArxivUrl] = useState("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [videos, setVideos] = useState<VideoItem[]>(mockVideos)
  const [faqs, setFaqs] = useState<VideoItem[]>(mockFaqs)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type === "application/pdf") {
      setSelectedFile(file)
      setArxivUrl("")
    }
  }

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setArxivUrl(e.target.value)
    if (e.target.value) {
      setSelectedFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  const handleSubmit = async () => {
    if (!arxivUrl && !selectedFile) return

    setIsSubmitting(true)
    setVideos([])
    setFaqs([])

    await new Promise((resolve) => setTimeout(resolve, 3000))

    for (let i = 0; i < mockVideos.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 300))
      setVideos((prev) => [...prev, mockVideos[i]])
    }

    for (let i = 0; i < mockFaqs.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 300))
      setFaqs((prev) => [...prev, mockFaqs[i]])
    }

    setIsSubmitting(false)
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
                />
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
                  {videos.length === 0 ? (
                    <div className="flex h-full items-center justify-center text-muted-foreground">
                      {isSubmitting ? (
                        <div className="flex flex-col items-center gap-2">
                          <Loader2 className="h-8 w-8 animate-spin" />
                          <p className="text-center text-sm">Generating videos...</p>
                        </div>
                      ) : (
                        <p className="text-center text-sm">Submit a paper to generate videos</p>
                      )}
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
                  {faqs.length === 0 ? (
                    <div className="flex h-full items-center justify-center text-muted-foreground">
                      {isSubmitting ? (
                        <div className="flex flex-col items-center gap-2">
                          <Loader2 className="h-8 w-8 animate-spin" />
                          <p className="text-center text-sm">Generating FAQ videos...</p>
                        </div>
                      ) : (
                        <p className="text-center text-sm">Submit a paper to generate FAQ videos</p>
                      )}
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
  return (
    <div className="flex gap-4 rounded-lg border bg-card p-3 transition-colors hover:bg-muted/50">
      <div className="relative flex-shrink-0">
        <img
          src={video.thumbnail || "/placeholder.svg"}
          alt={video.title}
          className="h-24 w-40 rounded-md object-cover"
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex h-10 w-10 cursor-pointer items-center justify-center rounded-full bg-background/80 backdrop-blur-sm transition-transform hover:scale-110">
            <Play className="h-5 w-5 text-foreground" />
          </div>
        </div>
      </div>
      <div className="flex flex-col justify-center overflow-hidden">
        <h3 className="font-semibold text-foreground">{video.title}</h3>
        <p className="line-clamp-3 text-sm text-muted-foreground">{video.description}</p>
      </div>
    </div>
  )
}
