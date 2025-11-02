from deepface import DeepFace
from pathlib import Path
from PIL import Image
import json
import numpy as np
from typing import Optional, Dict, List, Any
import hashlib
import tempfile
import requests
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FaceRecognitionService:
    THRESHOLDS = {
        'VGG-Face': 0.40,
        'Facenet': 0.40,
        'Facenet512': 0.30,
        'OpenFace': 0.10,
        'DeepFace': 0.23,
        'DeepID': 0.015,
        'ArcFace': 0.68,
        'Dlib': 0.07,
        'SFace': 0.593
    }
    
    def __init__(
        self, 
        model_name: str = "VGG-Face",
        distance_metric: str = "cosine",
        enforce_detection: bool = False,
        detector_backend: str = "opencv",
        cache_images: bool = True,
        extract_faces: bool = True, 
        align_faces: bool = True,     
        expand_face_region: float = 1.2  
    ):
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.enforce_detection = enforce_detection
        self.detector_backend = detector_backend
        self.cache_images = cache_images
        self.extract_faces = extract_faces
        self.align_faces = align_faces
        self.expand_face_region = expand_face_region
        self.image_cache = {}
        self.face_cache = {}  # Cache for extracted faces
    
    def verify_image(self, image_path: str) -> bool:
        try:
            if not Path(image_path).exists():
                logger.debug(f"Image file not found: {image_path}")
                return False
            
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        rgb_path = Path(image_path).with_suffix('.jpg')
                        img.convert('RGB').save(rgb_path, 'JPEG')
            except Exception as e:
                logger.debug(f"Failed to open image: {e}")
                return False
            
            return True
        except Exception as e:
            logger.debug(f"Image verification failed: {e}")
            return False
    
    def download_and_cache_image(self, url: str) -> Optional[str]:
        if not url:
            return None
            
        if not url.startswith('http'):
            return url if self.verify_image(url) else None
        
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        if self.cache_images and url_hash in self.image_cache:
            cached_path = self.image_cache[url_hash]
            if self.verify_image(cached_path):
                return cached_path
        
        try:
            logger.debug(f"Downloading: {url}")
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            
            temp_file.close()
            temp_path = temp_file.name

            if self.verify_image(temp_path):
                if self.cache_images:
                    self.image_cache[url_hash] = temp_path
                return temp_path
            else:
                return None
            
        except Exception as e:
            logger.debug(f"Failed to download image from {url}: {e}")
            return None
    
    def extract_face_from_image(
        self, 
        image_path: str,
        target_size: tuple = (224, 224),
        return_largest: bool = True
    ) -> Optional[str]:

        cache_key = hashlib.md5(f"{image_path}_extracted".encode()).hexdigest()
        
        if self.cache_images and cache_key in self.face_cache:
            cached_path = self.face_cache[cache_key]
            if self.verify_image(cached_path):
                logger.debug(f"Using cached extracted face")
                return cached_path
        
        try:
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Don't fail if no face detected
                align=self.align_faces,
                target_size=target_size
            )
            
            if not face_objs:
                logger.debug("No faces detected in image")
                return None
            
            if len(face_objs) > 1:
                logger.debug(f"Found {len(face_objs)} faces in image")
                if return_largest:
                    # Get the face with largest area
                    face_obj = max(face_objs, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                    logger.debug(f"Selected largest face")
                else:
                    # Return the first face
                    face_obj = face_objs[0]
            else:
                face_obj = face_objs[0]

            face_img = face_obj['face']

            if face_img.max() <= 1.0:
                face_img = (face_img * 255).astype(np.uint8)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            face_pil = Image.fromarray(face_img)
            face_pil.save(temp_file.name, 'JPEG', quality=95)
            temp_file.close()
            
            logger.debug(f"Extracted face saved to {temp_file.name}")

            if self.cache_images:
                self.face_cache[cache_key] = temp_file.name
            
            return temp_file.name
            
        except Exception as e:
            logger.debug(f"Face extraction failed: {str(e)[:100]}")
            return None
    
    def extract_face_with_expansion(
        self,
        image_path: str,
        target_size: tuple = (224, 224)
    ) -> Optional[str]:
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect face
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False  # Don't align yet, we need original coordinates
            )
            
            if not face_objs:
                return None
            
            # Get largest face
            face_obj = max(face_objs, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
            facial_area = face_obj['facial_area']
            
            # Expand the bounding box
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            # Calculate expansion
            expand_w = int(w * (self.expand_face_region - 1) / 2)
            expand_h = int(h * (self.expand_face_region - 1) / 2)
            
            # New coordinates with bounds checking
            x1 = max(0, x - expand_w)
            y1 = max(0, y - expand_h)
            x2 = min(img_rgb.shape[1], x + w + expand_w)
            y2 = min(img_rgb.shape[0], y + h + expand_h)
            
            # Extract expanded face region
            face_expanded = img_rgb[y1:y2, x1:x2]
            
            # Resize to target size
            face_resized = cv2.resize(face_expanded, target_size)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            face_pil = Image.fromarray(face_resized)
            face_pil.save(temp_file.name, 'JPEG', quality=95)
            temp_file.close()
            
            logger.debug(f"Extracted expanded face ({self.expand_face_region}x)")
            return temp_file.name
            
        except Exception as e:
            logger.debug(f"Expanded face extraction failed: {str(e)[:100]}")
            return None
    
    def preprocess_image(self, image_path: str) -> str:

        if not self.extract_faces:
            return image_path
        
        # Try to extract face
        extracted_path = self.extract_face_from_image(image_path)
        
        if extracted_path:
            logger.debug(f"Using extracted face for comparison")
            return extracted_path
        else:
            logger.debug(f"No face extracted, using original image")
            return image_path
    
    def load_linkedin_profiles(self, json_file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(json_file_path, 'r') as f:
                profiles = json.load(f)
            
            # Standardize the profile format
            standardized_profiles = []
            for profile in profiles:
                standardized = {
                    "name": profile.get("fullName", "Unknown"),
                    "profileUrl": f"https://linkedin.com/in/{profile.get('public_identifier', '')}",
                    "imageUrl": profile.get("profile_photo", ""),
                    "linkedin_id": profile.get("linkedin_internal_id", ""),
                    "headline": profile.get("headline", ""),
                    "location": profile.get("location", ""),
                    "connections": profile.get("connections", ""),
                    "about": profile.get("about", ""),
                    "experience": profile.get("experience", []),
                    "education": profile.get("education", []),
                    "raw_profile": profile
                }
                standardized_profiles.append(standardized)
            
            logger.info(f"Loaded {len(standardized_profiles)} profiles")
            return standardized_profiles
            
        except Exception as e:
            logger.error(f"Error loading LinkedIn profiles: {e}")
            return []
    
    def calculate_confidence(self, distance: float, model_name: str) -> float:
        """Calculate confidence score from distance."""
        threshold = self.THRESHOLDS.get(model_name, 0.40)
        
        # Simple but effective: confidence decreases as distance increases
        if distance < threshold:
            confidence = 1.0 - (distance / threshold) * 0.5
        else:
            confidence = max(0, 0.5 - (distance - threshold) / threshold * 0.5)
        
        return max(0, min(1.0, confidence))
    
    def compare_faces(
        self,
        img1_path: str,
        img2_path: str
    ) -> Optional[Dict[str, Any]]:
        """Compare two faces and return distance and confidence."""
        try:
            # Preprocess images (extract faces if enabled)
            processed_img1 = self.preprocess_image(img1_path)
            processed_img2 = self.preprocess_image(img2_path)
            
            result = DeepFace.verify(
                img1_path=processed_img1,
                img2_path=processed_img2,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend
            )
            
            distance = result['distance']
            confidence = self.calculate_confidence(distance, self.model_name)
            
            return {
                'distance': distance,
                'confidence': confidence,
                'verified': result.get('verified', False),
                'threshold': result.get('threshold', self.THRESHOLDS.get(self.model_name, 0.40)),
                'face_extracted': self.extract_faces
            }
            
        except Exception as e:
            logger.debug(f"Face comparison failed: {str(e)[:100]}")
            return None
    
    def find_best_match(
        self,
        target_image_path: str,
        profiles_json_path: str,
        image_field: str = "imageUrl",
        min_confidence: float = 0.0,
        return_top_n: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching profile(s)."""
        
        # Verify target image
        if not self.verify_image(target_image_path):
            logger.error("❌ Target image verification failed")
            return None
        
        logger.info(f"✓ Target image verified: {target_image_path}")
        
        # If face extraction is enabled, show what we found
        if self.extract_faces:
            extracted = self.extract_face_from_image(target_image_path)
            if extracted:
                logger.info(f"✓ Face extracted from target image")
            else:
                logger.info(f"⚠️  No face extracted, will use full image")
        
        # Load profiles
        profiles = self.load_linkedin_profiles(profiles_json_path)
        if not profiles:
            logger.error("❌ No profiles loaded")
            return None
        
        matches = []
        processed_count = 0
        error_count = 0
        face_extracted_count = 0
        
        for i, profile in enumerate(profiles):
            if image_field not in profile or not profile[image_field]:
                continue
            
            image_url = profile[image_field]
            name = profile.get('name', 'Unknown')
            
            logger.info(f"\n[{i+1}/{len(profiles)}] Processing: {name}")
            
            # Download and cache profile image
            profile_img_path = self.download_and_cache_image(image_url)
            
            if not profile_img_path:
                logger.warning(f"  ⚠️  Could not load image")
                error_count += 1
                continue
            
            try:
                # Compare faces
                result = self.compare_faces(target_image_path, profile_img_path)
                
                if result:
                    matches.append({
                        'profile': profile.copy(),
                        'distance': result['distance'],
                        'confidence': result['confidence'],
                        'verified': result['verified'],
                        'threshold': result['threshold'],
                        'face_extracted': result.get('face_extracted', False)
                    })
                    
                    if result.get('face_extracted'):
                        face_extracted_count += 1
                    
                    logger.info(f"  ✓ Distance: {result['distance']:.4f}, Confidence: {result['confidence']:.2%}")
                    processed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.warning(f"  ⚠️  Error: {str(e)[:100]}")
                error_count += 1
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processed: {processed_count}/{len(profiles)} profiles")
        logger.info(f"Errors: {error_count}")
        if self.extract_faces:
            logger.info(f"Faces extracted: {face_extracted_count}/{processed_count}")
        
        if not matches:
            logger.warning("❌ No valid comparisons completed")
            return None
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Filter by minimum confidence
        if min_confidence > 0:
            matches = [m for m in matches if m['confidence'] >= min_confidence]
            logger.info(f"Matches above {min_confidence:.0%} confidence: {len(matches)}")
        
        if not matches:
            logger.warning(f"❌ No matches above {min_confidence:.0%} confidence threshold")
            return None
        
        # Show top matches
        logger.info(f"\n{'='*60}")
        logger.info(f"TOP {min(return_top_n, len(matches))} MATCH(ES)")
        logger.info(f"{'='*60}")
        
        for idx, match in enumerate(matches[:return_top_n], 1):
            profile = match['profile']
            logger.info(f"\n#{idx} - {profile.get('name', 'Unknown')}")
            logger.info(f"    Headline: {profile.get('headline', 'N/A')}")
            logger.info(f"    Confidence: {match['confidence']:.2%}")
            logger.info(f"    Distance: {match['distance']:.4f}")
            logger.info(f"    Threshold: {match['threshold']:.4f}")
            logger.info(f"    Verified: {'✓' if match['verified'] else '✗'}")
        
        # Return best match or top N
        if return_top_n == 1:
            best = matches[0]
            return {
                "matched_profile": best['profile'],
                "confidence": best['confidence'],
                "face_distance": best['distance'],
                "verified": best['verified']
            }
        else:
            return {
                "matches": [
                    {
                        "matched_profile": m['profile'],
                        "confidence": m['confidence'],
                        "face_distance": m['distance'],
                        "verified": m['verified']
                    }
                    for m in matches[:return_top_n]
                ],
                "total_found": len(matches)
            }
    
    def find_all_matches(
        self,
        target_image_path: str,
        profiles_json_path: str,
        image_field: str = "imageUrl",
        max_results: int = 5,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Find all matching profiles ranked by similarity."""
        
        result = self.find_best_match(
            target_image_path=target_image_path,
            profiles_json_path=profiles_json_path,
            image_field=image_field,
            min_confidence=min_confidence,
            return_top_n=max_results
        )
        
        if not result:
            return []
        
        if 'matches' in result:
            return result['matches']
        else:
            return [result]


def main(json_data, target_img_data, **kwargs):
    
    # Extract kwargs with defaults
    model_name = kwargs.get('model_name', 'VGG-Face')
    enforce_detection = kwargs.get('enforce_detection', False)
    min_confidence = kwargs.get('min_confidence', 0.0)
    max_results = kwargs.get('max_results', 5)
    extract_faces = kwargs.get('extract_faces', True)  # NEW
    align_faces = kwargs.get('align_faces', True)      # NEW
    expand_face_region = kwargs.get('expand_face_region', 1.2)  # NEW
    
    service = FaceRecognitionService(
        model_name=model_name,
        distance_metric="cosine",
        enforce_detection=enforce_detection,
        cache_images=True,
        extract_faces=extract_faces,
        align_faces=align_faces,
        expand_face_region=expand_face_region
    )
    
    # Handle json_data
    if isinstance(json_data, str):
        profiles_json_path = json_data
    elif isinstance(json_data, list):
        temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(json_data, temp_json)
        temp_json.close()
        profiles_json_path = temp_json.name
    else:
        raise ValueError("json_data must be either a file path (str) or list of profiles")
    
    # Handle target_img_data
    if isinstance(target_img_data, str):
        target_image_path = target_img_data
    else:
        temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        
        if isinstance(target_img_data, Image.Image):
            target_img_data.save(temp_img.name, 'JPEG')
        elif isinstance(target_img_data, np.ndarray):
            Image.fromarray(target_img_data).save(temp_img.name, 'JPEG')
        else:
            raise ValueError("target_img_data must be a file path, PIL Image, or numpy array")
        
        target_image_path = temp_img.name

    # Run the matching
    results = service.find_all_matches(
        target_image_path=target_image_path,
        profiles_json_path=profiles_json_path,
        image_field="imageUrl",
        max_results=max_results,
        min_confidence=min_confidence
    )

    return results


# Example usage with different configurations:



import json
from PIL import Image

profiles = json.load(open("./docs/sample_output.json"))
img = Image.open("./face_recognition_service/test_images/test_img3.jpg")

results = main(profiles, img, model_name="Facenet512", extract_faces=True, min_confidence=0.1)
print(results[0])
