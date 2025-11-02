# Face Recognition Feature

This document describes the face recognition functionality added to the Opp Trace dashboard.

## Overview

The face recognition feature allows users to take a photo using their webcam and automatically match it against LinkedIn profile photos of attendees. This is useful for quickly identifying attendees at events.

## Architecture

### Components

1. **Python Face Recognition Service** (`website/lib/faceRecognitionService.py`)
   - Uses DeepFace library with VGG-Face model
   - Compares target face against all LinkedIn profile photos
   - Returns best match with confidence score

2. **Next.js API Route** (`website/app/api/match-face/route.ts`)
   - Accepts base64 encoded image from frontend
   - Calls Python subprocess to perform face matching
   - Returns matched profile with confidence score

3. **Camera Modal Component** (`website/components/CameraModal.tsx`)
   - Requests webcam access via browser API
   - Displays live video preview
   - Captures photo and converts to base64

4. **Match Result Component** (`website/components/MatchResult.tsx`)
   - Displays matched profile information
   - Shows confidence percentage
   - Displays LinkedIn data, scores, and social links

## How It Works

### Data Flow

```
User clicks "Find by Face" button
        ↓
Camera Modal opens → User takes photo
        ↓
Photo converted to base64 → Sent to API
        ↓
API writes attendees JSON to temp file
        ↓
Python subprocess spawned → Face matching
        ↓
DeepFace compares target vs all profile photos
        ↓
Best match returned to frontend
        ↓
Dashboard highlights matched row
        ↓
Match Result modal displays profile details
```

### Face Recognition Algorithm

- **Model**: VGG-Face (96.7% accuracy)
- **Distance Metric**: Cosine similarity
- **Threshold**: Returns best match regardless of distance
- **Detection**: `enforce_detection=False` for better matching
- **Confidence**: Calculated as `1 - distance` (0-1 scale)

### Performance

- **Processing Time**: 2-5 seconds per comparison
- **Total Time**: Depends on number of attendees with photos
- **Optimization**: Processes all comparisons in sequence
- **Timeout**: 30 seconds maximum

## Usage

### Prerequisites

1. Attendees must have LinkedIn data scraped (including profile photos)
2. User must grant webcam permissions
3. Python environment with DeepFace installed

### Step-by-Step

1. **Navigate to Dashboard**
   - Ensure attendees have been scraped and LinkedIn data is available
   - Wait for scraping progress to complete

2. **Click "Find by Face" Button**
   - Located in the summary section next to "Download CSV"
   - Triggers camera modal to open

3. **Take Photo**
   - Allow camera permissions when prompted
   - Position face in camera view
   - Click "Capture Photo" button
   - Camera automatically closes after capture

4. **Wait for Matching**
   - Loading indicator shows "Analyzing face and matching with attendees..."
   - Typically takes 2-10 seconds depending on number of attendees

5. **View Results**
   - If match found:
     - Matched row highlights with green gradient
     - Row scrolls into view automatically
     - Match Result modal displays with:
       - Confidence percentage (e.g., "94.2% Match")
       - Profile photo and basic info
       - LinkedIn experience and education
       - Hackathon scores (if available)
   - If no match:
     - Error message displays above table
     - Can retry by clicking "Find by Face" again

## Configuration

### Model Selection

In `website/lib/faceRecognitionService.py`, you can change the model:

```python
service = FaceRecognitionService(
    model_name="VGG-Face",      # Options: VGG-Face, Facenet512, ArcFace, etc.
    distance_metric="cosine"     # Options: cosine, euclidean, euclidean_l2
)
```

**Available Models** (by accuracy):
- Facenet512: 98.4% (slower, most accurate)
- Facenet: 97.4%
- Dlib: 96.8%
- VGG-Face: 96.7% (default - good balance)
- ArcFace: 96.7%

### Timeout

In `website/app/api/match-face/route.ts`, adjust the timeout:

```typescript
setTimeout(() => {
  pythonProcess.kill();
  reject(new Error('Face recognition timeout'));
}, 30000);  // 30 seconds (adjust as needed)
```

## Troubleshooting

### Common Issues

**1. "Camera access denied"**
- Grant camera permissions in browser settings
- Check browser console for specific error
- Try HTTPS connection (some browsers require secure context)

**2. "No attendees with LinkedIn profile photos found"**
- Ensure LinkedIn scraping has completed
- Check that attendees have `linkedinData.profile_photo` populated
- Verify ScrapingDog API key is configured

**3. "Face recognition timeout"**
- Reduce number of attendees to compare against
- Increase timeout value in API route
- Check Python subprocess is working correctly

**4. "No matching face found"**
- Ensure target face is clearly visible in photo
- Verify LinkedIn profile photos are accessible
- Try with different lighting conditions
- Consider lowering detection strictness

**5. Python subprocess errors**
- Ensure Python 3 is installed and in PATH
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check `website/temp/` directory exists and is writable

### Debugging

Enable detailed logging in Python service:

```python
# In faceRecognitionService.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check Next.js console output:

```bash
cd website
npm run dev
# Watch for Python stderr output
```

## Security Considerations

1. **Camera Access**: Requires user permission via browser API
2. **Image Data**: Not stored permanently (only in temp files, cleaned up immediately)
3. **Privacy**: Face data only compared locally, not sent to external services
4. **Permissions**: Ensure proper file permissions for `website/temp/` directory

## Future Enhancements

Potential improvements:

1. **Batch Upload**: Allow uploading multiple photos at once
2. **Confidence Threshold**: Only show matches above certain confidence
3. **Multiple Matches**: Show top 3 closest matches
4. **Face Detection**: Validate face exists before processing
5. **GPU Acceleration**: Use TensorFlow GPU support for faster processing
6. **Database Storage**: Cache embeddings for faster repeated searches
7. **Real-time Video**: Match faces in real-time video stream

## API Reference

### POST `/api/match-face`

**Request Body:**
```json
{
  "imageData": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Success Response:**
```json
{
  "success": true,
  "match": {
    "profile": {
      "name": "John Doe",
      "profileUrl": "https://lu.ma/user/...",
      "linkedinData": { ... },
      "overall_score": 85,
      ...
    },
    "confidence": 0.942,
    "distance": 0.058,
    "verified": true
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "No matching face found"
}
```

## Technical Details

### Dependencies

**Python:**
- `deepface==0.0.95` - Face recognition framework
- `tensorflow==2.20.0` - Deep learning backend
- `opencv-python==4.12.0.88` - Computer vision
- `pillow==12.0.0` - Image processing
- `mtcnn==1.0.0` - Face detection

**JavaScript:**
- React hooks for state management
- Browser MediaDevices API for camera access
- Canvas API for image capture

### File Structure

```
website/
├── lib/
│   └── faceRecognitionService.py    # Python face matching logic
├── app/
│   ├── api/
│   │   └── match-face/
│   │       └── route.ts              # API endpoint
│   ├── page.tsx                      # Dashboard with face matching
│   └── page.module.css               # Styles including highlights
├── components/
│   ├── CameraModal.tsx               # Camera capture UI
│   ├── CameraModal.module.css
│   ├── MatchResult.tsx               # Match result display
│   └── MatchResult.module.css
└── temp/                             # Temporary files (gitignored)
    ├── attendees_*.json              # Auto-generated, cleaned up
    └── target_face.jpg               # Auto-generated, cleaned up
```

## Performance Optimization

### Current Approach
- Sequential comparison (one profile at a time)
- Creates new temp files for each request
- Spawns new Python process each time

### Potential Optimizations
1. **Cache Embeddings**: Pre-compute face embeddings for all profiles
2. **Persistent Process**: Keep Python process running for multiple requests
3. **Parallel Processing**: Compare multiple profiles simultaneously
4. **Vector Database**: Use FAISS or similar for fast similarity search

## References

- [DeepFace GitHub](https://github.com/serengil/deepface)
- [VGG-Face Paper](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/)
- [MediaDevices API](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices)
- [Face Recognition Best Practices](https://en.wikipedia.org/wiki/Facial_recognition_system)
