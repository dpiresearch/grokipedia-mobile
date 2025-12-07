const fs = require('fs');
const path = require('path');

// Copy paper-manim-viz-explanations directory into frontend-mobile for Vercel deployment
const sourceDir = path.resolve(__dirname, '../../paper-manim-viz-explanations');
const destDir = path.resolve(__dirname, '../paper-manim-viz-explanations');

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

if (fs.existsSync(sourceDir)) {
  console.log(`Copying ${sourceDir} to ${destDir}...`);
  copyRecursiveSync(sourceDir, destDir);
  console.log('Assets copied successfully!');
} else {
  console.warn(`Warning: Source directory ${sourceDir} does not exist. Skipping asset copy.`);
}

