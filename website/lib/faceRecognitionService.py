#!/usr/bin/env python3
"""
Face Recognition Service for Opp Trace Dashboard
Matches a target face image against LinkedIn profile photos using DeepFace
"""

import sys
import json
import base64
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from deepface import DeepFace
from PIL import Image
import io

class FaceRecognitionService:
    """Service for matching faces against LinkedIn profile photos"""

    def __init__(self, model_name: str = "VGG-Face", distance_metric: str = "cosine"):
        """
        Initialize face recognition service

        Args:
            model_name: DeepFace model to use (VGG-Face, Facenet, ArcFace, etc.)
            distance_metric: Distance metric (cosine, euclidean, euclidean_l2)
        """
        self.model_name = model_name
        self.distance_metric = distance_metric

    def decode_base64_image(self, base64_string: str, output_path: str) -> bool:
        """
        Decode base64 image string and save to file

        Args:
            base64_string: Base64 encoded image (with or without data URI prefix)
            output_path: Path to save decoded image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove data URI prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            # Decode base64 to bytes
            image_data = base64.b64decode(base64_string)

            # Open and convert image to RGB (DeepFace requirement)
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Save as JPEG
            image.save(output_path, 'JPEG')
            return True

        except Exception as e:
            print(f"Error decoding base64 image: {str(e)}", file=sys.stderr)
            return False

    def verify_image(self, image_path: str) -> bool:
        """
        Verify that an image file is valid and readable

        Args:
            image_path: Path to image file

        Returns:
            True if valid, False otherwise
        """
        try:
            if not os.path.exists(image_path):
                return False

            # Try to open with PIL
            with Image.open(image_path) as img:
                img.verify()

            return True

        except Exception:
            return False

    def download_image(self, url: str, output_path: str) -> bool:
        """
        Download an image from a URL and save to local file

        Args:
            url: URL of the image to download
            output_path: Path where to save the downloaded image

        Returns:
            True if download successful, False otherwise
        """
        try:
            # Add headers to mimic browser request (some sites block without User-Agent)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            # Download with timeout
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)

            if response.status_code == 200:
                # Verify it's actually an image by trying to open it
                try:
                    image = Image.open(io.BytesIO(response.content))
                    # Convert to RGB if needed (JPEG requirement)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    # Save as JPEG
                    image.save(output_path, 'JPEG')
                    return True
                except Exception as e:
                    print(f"Error processing downloaded image: {str(e)}", file=sys.stderr)
                    return False
            else:
                print(f"Failed to download image: HTTP {response.status_code}", file=sys.stderr)
                return False

        except requests.exceptions.Timeout:
            print(f"Timeout downloading image from: {url}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error downloading image from {url}: {str(e)}", file=sys.stderr)
            return False

    def match_face(
        self,
        target_image_path: str,
        profiles: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best matching profile for a target face image

        Args:
            target_image_path: Path to the target face image
            profiles: List of profile dictionaries with LinkedIn data

        Returns:
            Dictionary with matched profile, confidence, and distance
            None if no valid match found
        """
        if not self.verify_image(target_image_path):
            print(f"Target image is not valid: {target_image_path}", file=sys.stderr)
            return None

        best_match = None
        best_distance = float('inf')

        # Create temp directory for downloaded profile photos
        temp_dir = Path(__file__).parent.parent / 'temp'
        temp_dir.mkdir(exist_ok=True)

        downloaded_files = []  # Track files to clean up

        print(f"Processing {len(profiles)} profiles for face matching...", file=sys.stderr)

        # Iterate through all profiles with photos
        for index, profile in enumerate(profiles):
            # Skip profiles without LinkedIn data or profile photos
            linkedin_data = profile.get('linkedinData')
            if not linkedin_data:
                continue

            profile_photo_url = linkedin_data.get('profile_photo')
            if not profile_photo_url:
                continue

            try:
                # Download profile photo from URL to temp file
                temp_photo_path = temp_dir / f'profile_{index}.jpg'

                print(f"[{index + 1}/{len(profiles)}] Downloading profile photo for {profile.get('name')}...", file=sys.stderr)

                if not self.download_image(profile_photo_url, str(temp_photo_path)):
                    print(f"Failed to download photo for {profile.get('name')}, skipping...", file=sys.stderr)
                    continue

                downloaded_files.append(temp_photo_path)

                print(f"[{index + 1}/{len(profiles)}] Comparing faces with {profile.get('name')}...", file=sys.stderr)

                try:
                    # Use DeepFace.verify to compare faces with local file
                    result = DeepFace.verify(
                        img1_path=target_image_path,
                        img2_path=str(temp_photo_path),  # Now using local file!
                        model_name=self.model_name,
                        distance_metric=self.distance_metric,
                        enforce_detection=False,  # More lenient for better matching
                        detector_backend='skip'  # Skip face detection preprocessing
                    )

                    distance = result['distance']

                    print(f"[{index + 1}/{len(profiles)}] Distance: {distance:.4f}, Verified: {result['verified']}", file=sys.stderr)

                    # Track the closest match
                    if distance < best_distance:
                        best_distance = distance
                        best_match = {
                            'matched_profile': profile,
                            'confidence': 1 - distance,  # Convert distance to confidence
                            'distance': distance,
                            'verified': result['verified']
                        }
                        print(f"✓ New best match: {profile.get('name')} (confidence: {(1-distance)*100:.1f}%)", file=sys.stderr)

                except ValueError as e:
                    # Face detection failed for this specific photo
                    error_msg = str(e)
                    if "Face could not be detected" in error_msg or "Face could not be found" in error_msg:
                        print(f"[{index + 1}/{len(profiles)}] ⚠️  No face detected in photo for {profile.get('name')}, skipping...", file=sys.stderr)
                    else:
                        print(f"[{index + 1}/{len(profiles)}] ValueError: {error_msg}", file=sys.stderr)
                    continue
                except Exception as e:
                    # Other errors during verification
                    print(f"[{index + 1}/{len(profiles)}] Verification error for {profile.get('name')}: {str(e)}", file=sys.stderr)
                    continue

            except Exception as e:
                # Download or general errors
                print(f"Error processing {profile.get('name')}: {str(e)}", file=sys.stderr)
                continue

        # Clean up downloaded profile photos
        print(f"Cleaning up {len(downloaded_files)} temporary files...", file=sys.stderr)
        for temp_file in downloaded_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                print(f"Error deleting temp file {temp_file}: {str(e)}", file=sys.stderr)

        if best_match:
            print(f"✓ Best match found: {best_match['matched_profile'].get('name')} with {best_match['confidence']*100:.1f}% confidence", file=sys.stderr)
        else:
            print(f"No matching face found among {len(profiles)} profiles", file=sys.stderr)

        return best_match


def main():
    """
    Main function to run face matching from command line

    Expected usage:
        python faceRecognitionService.py <base64_image> <attendees_json_path>

    Output:
        JSON object with match results
    """
    if len(sys.argv) != 3:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python faceRecognitionService.py <base64_image> <attendees_json_path>'
        }))
        sys.exit(1)

    base64_image = sys.argv[1]
    attendees_json_path = sys.argv[2]

    # Create temp directory if it doesn't exist
    temp_dir = Path(__file__).parent.parent / 'temp'
    temp_dir.mkdir(exist_ok=True)

    # Decode base64 image to temporary file
    target_image_path = temp_dir / 'target_face.jpg'

    service = FaceRecognitionService(model_name="VGG-Face", distance_metric="cosine")

    if not service.decode_base64_image(base64_image, str(target_image_path)):
        print(json.dumps({
            'success': False,
            'error': 'Failed to decode base64 image'
        }))
        sys.exit(1)

    # Load attendees data
    try:
        with open(attendees_json_path, 'r') as f:
            attendees_data = json.load(f)
            attendees = attendees_data.get('attendees', [])
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Failed to load attendees JSON: {str(e)}'
        }))
        sys.exit(1)

    # Find best match
    match_result = service.match_face(str(target_image_path), attendees)

    # Clean up temporary image
    try:
        target_image_path.unlink()
    except:
        pass

    # Output result as JSON
    if match_result:
        print(json.dumps({
            'success': True,
            'match': {
                'profile': match_result['matched_profile'],
                'confidence': match_result['confidence'],
                'distance': match_result['distance'],
                'verified': match_result['verified']
            }
        }))
    else:
        print(json.dumps({
            'success': False,
            'error': 'No matching face found'
        }))


if __name__ == '__main__':
    main()
