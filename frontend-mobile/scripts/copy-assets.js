const fs = require('fs');
const path = require('path');

// Copy paper-manim-viz-explanations directory into frontend-mobile for Vercel deployment
// Try multiple possible source locations to handle both local and Vercel builds
const possibleSourceDirs = [
  path.resolve(__dirname, '../../paper-manim-viz-explanations'), // Local dev: from frontend-mobile/scripts
  path.resolve(process.cwd(), '../paper-manim-viz-explanations'), // Vercel: if root is frontend-mobile
  path.resolve(process.cwd(), 'paper-manim-viz-explanations'), // Already in place
];

const destDir = path.resolve(process.cwd(), 'paper-manim-viz-explanations');

function copyRecursiveSync(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();

  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach(childItemName => {
      copyRecursiveSync(
        path.join(src, childItemName),
        path.join(dest, childItemName)
      );
    });
  } else {
    fs.copyFileSync(src, dest);
  }
}

// Find the first existing source directory
let sourceDir = null;
for (const possibleDir of possibleSourceDirs) {
  if (fs.existsSync(possibleDir)) {
    sourceDir = possibleDir;
    break;
  }
}

if (sourceDir) {
  console.log(`[Build] Found source directory: ${sourceDir}`);
  console.log(`[Build] Copying to: ${destDir}`);
  
  // Remove destination if it exists to ensure clean copy
  if (fs.existsSync(destDir)) {
    console.log(`[Build] Removing existing destination directory...`);
    fs.rmSync(destDir, { recursive: true, force: true });
  }
  
  try {
    copyRecursiveSync(sourceDir, destDir);
    
    // Verify the copy was successful
    if (fs.existsSync(destDir)) {
      const stats = fs.statSync(destDir);
      if (stats.isDirectory()) {
        const fileCount = fs.readdirSync(destDir).length;
        console.log(`[Build] Assets copied successfully! (${fileCount} top-level items)`);
      } else {
        throw new Error('Destination exists but is not a directory');
      }
    } else {
      throw new Error('Destination directory was not created');
    }
  } catch (error) {
    console.error(`[Build] Error during copy: ${error.message}`);
    throw error;
  }
} else {
  console.warn(`[Build] WARNING: Source directory not found. Tried:`);
  possibleSourceDirs.forEach(dir => {
    console.warn(`  - ${dir} (exists: ${fs.existsSync(dir)})`);
  });
  console.warn(`[Build] Current working directory: ${process.cwd()}`);
  console.warn(`[Build] Script directory: ${__dirname}`);
  console.warn(`[Build] This is expected if paper-manim-viz-explanations is not in the repository.`);
  console.warn(`[Build] Continuing build, but API routes may not work without the assets.`);
  // Don't exit with error - allow build to continue for now
}

