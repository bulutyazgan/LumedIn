# Quick Installation Guide

## Step 1: Generate Icons

1. Open `create-icons.html` in your browser (double-click the file)
2. Click "Generate Icons" button
3. Three PNG files will download: `icon16.png`, `icon48.png`, `icon128.png`
4. Move these files into the `extension` folder

## Step 2: Install in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked"
4. Select the `extension` folder
5. Done! The extension is now installed

## Step 3: Use the Extension

1. Go to any Luma event page (e.g., `https://lu.ma/...`)
2. Look for the purple "Conduct Analysis" button
3. Click it to analyze attendees
4. A CSV file will download automatically

## Troubleshooting

**Button not showing?**
- Make sure you're on a Luma event page (URL has `/calendar/` and `/event/`)
- Refresh the page
- Check that the extension is enabled at `chrome://extensions/`

**Extension won't load?**
- Make sure all three icon files are in the extension folder
- Check the console at `chrome://extensions/` for error messages

**Need help?**
See the full README.md for detailed documentation.
