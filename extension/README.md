# Luma Attendee Analyzer - Chrome Extension

A Chrome extension that helps you analyze Luma event attendees by extracting their profiles and social media links, then exporting the data to CSV.

## Features

- Adds a "Conduct Analysis" button to Luma event pages
- Automatically scrapes attendee profiles from the guest list
- Extracts social media links (Instagram, X/Twitter, TikTok, LinkedIn, personal websites)
- Exports all data to a CSV file for easy analysis
- Batch processing with rate limiting to be respectful to Luma's servers

## Installation

### Method 1: Load Unpacked Extension (Development Mode)

1. **Download the extension**
   - Clone or download this repository
   - Navigate to the `extension` folder

2. **Add placeholder icons** (required by Chrome)
   - You need to add three icon files to the `extension` folder:
     - `icon16.png` (16x16 pixels)
     - `icon48.png` (48x48 pixels)
     - `icon128.png` (128x128 pixels)
   - You can create simple icons using any image editor or online tools

3. **Open Chrome Extensions page**
   - Open Google Chrome
   - Navigate to `chrome://extensions/`
   - Or click the three-dot menu → Extensions → Manage Extensions

4. **Enable Developer Mode**
   - Toggle the "Developer mode" switch in the top-right corner

5. **Load the extension**
   - Click "Load unpacked"
   - Select the `extension` folder from this project
   - The extension should now appear in your extensions list

6. **Verify installation**
   - Look for "Luma Attendee Analyzer" in your extensions
   - Make sure it's enabled (toggle should be blue)

## Usage

1. **Navigate to a Luma event page**
   - Go to any Luma event URL (e.g., `https://lu.ma/calendar/evt-xyz`)
   - Make sure you're logged into Luma to see the guest list

2. **Find the "Conduct Analysis" button**
   - The extension will automatically inject a purple "Conduct Analysis" button on the page
   - The button should appear near the event header

3. **Run the analysis**
   - Click the "Conduct Analysis" button
   - The button will show "Analyzing..." while processing
   - Watch the browser console (F12 → Console) to see progress

4. **Download results**
   - Once complete, a CSV file will automatically download
   - Filename format: `luma_attendees_YYYY-MM-DD.csv`
   - The button will show "Analysis Complete!" for 3 seconds

5. **Review the CSV**
   - Open the downloaded CSV in Excel, Google Sheets, or any spreadsheet app
   - Columns: Name, Profile URL, Instagram, X, TikTok, LinkedIn, Website

## CSV Output Format

The exported CSV contains the following columns:

| Column | Description |
|--------|-------------|
| Name | Attendee's display name on Luma |
| Profile URL | Link to their Luma profile |
| Instagram | Instagram profile URL (if available) |
| X | X/Twitter profile URL (if available) |
| TikTok | TikTok profile URL (if available) |
| LinkedIn | LinkedIn profile URL (if available) |
| Website | Personal website URL (if available) |

## Technical Details

### How It Works

1. **Button Injection**: The content script runs on all `lu.ma` pages and injects a styled button
2. **Guest List Access**: Clicks the guests button programmatically to open the attendee list
3. **Profile Scraping**: Fetches each attendee's Luma profile page to extract social links
4. **Batch Processing**: Processes 10 attendees at a time with 200ms delays between batches
5. **CSV Generation**: Formats data and triggers a browser download

### Permissions

The extension requires:
- `activeTab`: To interact with the current Luma page
- `https://lu.ma/*`: To access Luma event and profile pages

### Files Structure

```
extension/
├── manifest.json       # Extension configuration (Manifest V3)
├── content.js          # Main script that runs on Luma pages
├── styles.css          # Button styling
├── icon16.png          # 16x16 icon (you need to add this)
├── icon48.png          # 48x48 icon (you need to add this)
├── icon128.png         # 128x128 icon (you need to add this)
└── README.md           # This file
```

## Troubleshooting

### Button doesn't appear
- Make sure you're on a Luma event page (URL contains `/calendar/` and `/event/`)
- Try refreshing the page
- Check the browser console (F12) for any errors

### "Guests button not found" error
- Ensure you're logged into Luma
- The event page must have a guests/attendees section
- Some private events may not show guest lists

### Analysis fails or hangs
- Check your internet connection
- Some profiles may be private or deleted
- Open the browser console to see detailed error messages
- The extension will skip failed profiles and continue

### CSV not downloading
- Check your browser's download settings
- Make sure pop-ups aren't blocked for lu.ma
- Check the Downloads folder for the file

## Privacy & Ethics

- This extension only scrapes publicly available information from Luma
- It respects Luma's servers with rate limiting (200ms delays between batches)
- All processing happens locally in your browser
- No data is sent to external servers
- Please use responsibly and in compliance with Luma's Terms of Service

## Limitations

- Only works on Luma event pages where you have access to the guest list
- LinkedIn profiles are extracted as links only (not scraped due to anti-bot measures)
- Requires manual installation (not available on Chrome Web Store)
- May break if Luma changes their HTML structure

## Future Improvements

- Add LinkedIn profile scraping (requires careful implementation due to anti-bot measures)
- Create actual icon files automatically
- Add progress bar UI instead of just button text changes
- Allow customization of batch size and delays
- Export to multiple formats (JSON, Excel)

## Contributing

Feel free to submit issues or pull requests if you find bugs or want to add features!

## License

This is a personal tool. Use at your own risk and in compliance with Luma's Terms of Service.
