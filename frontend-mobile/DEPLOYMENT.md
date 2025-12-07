# Deployment Guide for Vercel

## Option 1: Deploy from Root Directory (Recommended)

1. Go to your Vercel dashboard
2. Import your repository
3. In project settings, configure:
   - **Root Directory**: `frontend-mobile`
   - **Framework Preset**: Next.js (auto-detected)
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)
   - **Install Command**: `npm install` (default)

Vercel will automatically detect Next.js and use the correct settings.

## Option 2: Deploy from Subdirectory

If deploying from the repository root:

1. Set **Root Directory** to `frontend-mobile`
2. All other settings will be auto-detected

## Environment Variables

No environment variables are required for basic functionality.

## File System Access

**Note**: This application reads files from the local filesystem (`paper-manim-viz-explanations` directory). For Vercel deployment, you have two options:

### Option A: Use Vercel's File System (Serverless Functions)

The API routes will work if the `paper-manim-viz-explanations` directory is included in your deployment. However, Vercel has limitations on file system access in serverless functions.

### Option B: Use External Storage (Recommended for Production)

For production, consider:
- Storing videos in a CDN (e.g., Cloudflare R2, AWS S3)
- Storing markdown files in a database or CMS
- Using API endpoints that fetch from external sources

## Build Configuration

The `next.config.mjs` is configured with:
- TypeScript build errors ignored (for faster builds)
- Unoptimized images (for compatibility)

**Note**: A prebuild script (`scripts/copy-assets.js`) automatically copies the `paper-manim-viz-explanations` directory into `frontend-mobile` during the build process. This ensures all assets are available for the API routes.

## Testing Locally

```bash
cd frontend-mobile
npm install
npm run dev
```

The app will run on `http://localhost:3001`

## Production Build

```bash
cd frontend-mobile
npm run build
npm start
```

