from deepface import DeepFace
from pathlib import Path
from PIL import Image, ImageEnhance
import json
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
import hashlib
import tempfile
import requests
import logging
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FaceRecognitionService:
    # Recommended thresholds for different models (cosine distance)
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
        detector_backend: str = "retinaface",  # Better detector
        cache_images: bool = True,
        extract_faces: bool = True,
        align_faces: bool = True,
        expand_face_region: float = 1.2,
        use_ensemble: bool = False,  # Use multiple models
        ensemble_models: List[str] = None,  # Custom ensemble
        enhance_images: bool = True,  # Image enhancement
        face_quality_threshold: float = 0.0,  # Minimum face quality
        use_parallel: bool = False,  # Parallel processing
        max_workers: int = 4
    ):
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.enforce_detection = enforce_detection
        self.detector_backend = detector_backend
        self.cache_images = cache_images
        self.extract_faces = extract_faces
        self.align_faces = align_faces
        self.expand_face_region = expand_face_region
        self.use_ensemble = use_ensemble
        self.enhance_images = enhance_images
        self.face_quality_threshold = face_quality_threshold
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        
        # Ensemble setup
        if ensemble_models:
            self.ensemble_models = ensemble_models
        elif use_ensemble:
            # Default high-accuracy ensemble
            self.ensemble_models = ['Facenet512', 'ArcFace', 'VGG-Face']
        else:
            self.ensemble_models = [model_name]
        
        self.image_cache = {}
        self.face_cache = {}
        self.embedding_cache = {}  # Cache embeddings for speed
    
    def verify_image(self, image_path: str) -> bool:
        """Quick verification that image exists and is valid."""
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
    
    def enhance_image_quality(self, image_path: str) -> str:
        """
        Enhance image quality for better face recognition.
        Applies brightness, contrast, and sharpness adjustments.
        """
        try:
            img = Image.open(image_path)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            
            # Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            
            # Save enhanced image
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(temp_file.name, 'JPEG', quality=95)
            temp_file.close()
            
            logger.debug("Image enhanced")
            return temp_file.name
            
        except Exception as e:
            logger.debug(f"Image enhancement failed: {e}")
            return image_path
    
    def assess_face_quality(self, face_obj: Dict) -> float:
        """
        Assess the quality of a detected face.
        Returns quality score between 0 and 1.
        """
        try:
            facial_area = face_obj.get('facial_area', {})
            confidence = face_obj.get('confidence', 0.5)
            
            # Check face size (larger is better)
            face_width = facial_area.get('w', 0)
            face_height = facial_area.get('h', 0)
            face_area = face_width * face_height
            
            # Normalize size score (assuming faces < 10000 pixels are small)
            size_score = min(1.0, face_area / 10000.0)
            
            # Detection confidence
            conf_score = confidence
            
            # Check aspect ratio (faces should be roughly square)
            if face_width > 0 and face_height > 0:
                aspect_ratio = face_width / face_height
                # Ideal ratio is around 1.0, penalize extremes
                aspect_score = 1.0 - abs(1.0 - aspect_ratio) * 0.5
                aspect_score = max(0, min(1.0, aspect_score))
            else:
                aspect_score = 0.5
            
            # Combined quality score
            quality = (size_score * 0.4 + conf_score * 0.4 + aspect_score * 0.2)
            
            return quality
            
        except Exception as e:
            logger.debug(f"Quality assessment failed: {e}")
            return 0.5
    
    def download_and_cache_image(self, url: str) -> Optional[str]:
        """Download image from URL and cache it locally."""
        if not url:
            return None
            
        if not url.startswith('http'):
            return url if self.verify_image(url) else None
        
        # Check cache first
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
    
    def extract_best_face(
        self, 
        image_path: str,
        target_size: tuple = (224, 224)
    ) -> Optional[Tuple[str, float]]:
        """
        Extract the best quality face from image.
        Returns (path, quality_score) or None.
        """
        cache_key = hashlib.md5(f"{image_path}_extracted".encode()).hexdigest()
        
        if self.cache_images and cache_key in self.face_cache:
            cached_path, quality = self.face_cache[cache_key]
            if self.verify_image(cached_path):
                logger.debug(f"Using cached extracted face (quality: {quality:.2f})")
                return cached_path, quality
        
        try:
            # Extract faces using better detector
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=self.align_faces,
                target_size=target_size
            )
            
            if not face_objs:
                logger.debug("No faces detected in image")
                return None
            
            # Select best quality face
            best_face = None
            best_quality = 0
            
            for face_obj in face_objs:
                quality = self.assess_face_quality(face_obj)
                if quality > best_quality:
                    best_quality = quality
                    best_face = face_obj
            
            if best_face is None or best_quality < self.face_quality_threshold:
                logger.debug(f"No face meets quality threshold ({best_quality:.2f} < {self.face_quality_threshold:.2f})")
                return None
            
            logger.debug(f"Selected face with quality: {best_quality:.2f}")
            
            # Get the face image
            face_img = best_face['face']
            
            if face_img.max() <= 1.0:
                face_img = (face_img * 255).astype(np.uint8)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            face_pil = Image.fromarray(face_img)
            face_pil.save(temp_file.name, 'JPEG', quality=95)
            temp_file.close()
            
            # Cache the result
            if self.cache_images:
                self.face_cache[cache_key] = (temp_file.name, best_quality)
            
            return temp_file.name, best_quality
            
        except Exception as e:
            logger.debug(f"Face extraction failed: {str(e)[:100]}")
            return None
    
    def preprocess_image(self, image_path: str) -> Tuple[str, float]:
        """
        Preprocess image with enhancement and face extraction.
        Returns (processed_path, quality_score).
        """
        # Enhance image if enabled
        if self.enhance_images:
            enhanced_path = self.enhance_image_quality(image_path)
        else:
            enhanced_path = image_path
        
        # Extract face if enabled
        if self.extract_faces:
            result = self.extract_best_face(enhanced_path)
            if result:
                return result
            else:
                logger.debug("Using original image (no face extracted)")
                return enhanced_path, 0.5
        else:
            return enhanced_path, 1.0
    
    def load_linkedin_profiles(self, json_file_path: str) -> List[Dict[str, Any]]:
        """Load LinkedIn profile data from JSON file."""
        try:
            with open(json_file_path, 'r') as f:
                profiles = json.load(f)
            
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
    
    def calculate_confidence(self, distance: float, model_name: str, quality1: float = 1.0, quality2: float = 1.0) -> float:
        """Calculate confidence score from distance and quality."""
        threshold = self.THRESHOLDS.get(model_name, 0.40)
        
        # Base confidence from distance
        if distance < threshold:
            base_confidence = 1.0 - (distance / threshold) * 0.5
        else:
            base_confidence = max(0, 0.5 - (distance - threshold) / threshold * 0.5)
        
        # Adjust by image quality (both images should be good quality)
        quality_factor = (quality1 + quality2) / 2.0
        adjusted_confidence = base_confidence * (0.7 + 0.3 * quality_factor)
        
        return max(0, min(1.0, adjusted_confidence))
    
    def compare_faces_single_model(
        self,
        img1_path: str,
        img2_path: str,
        model_name: str,
        quality1: float = 1.0,
        quality2: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Compare two faces using a single model."""
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=model_name,
                distance_metric=self.distance_metric,
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend
            )
            
            distance = result['distance']
            confidence = self.calculate_confidence(distance, model_name, quality1, quality2)
            
            return {
                'distance': distance,
                'confidence': confidence,
                'verified': result.get('verified', False),
                'threshold': result.get('threshold', self.THRESHOLDS.get(model_name, 0.40)),
                'model': model_name
            }
            
        except Exception as e:
            logger.debug(f"Comparison failed with {model_name}: {str(e)[:100]}")
            return None
    
    def compare_faces_ensemble(
        self,
        img1_path: str,
        img2_path: str,
        quality1: float = 1.0,
        quality2: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Compare faces using ensemble of models for better accuracy."""
        results = []
        
        for model in self.ensemble_models:
            result = self.compare_faces_single_model(img1_path, img2_path, model, quality1, quality2)
            if result:
                results.append(result)
        
        if not results:
            return None
        
        # Weighted voting based on model reliability
        weights = {
            'Facenet512': 1.5,
            'ArcFace': 1.5,
            'VGG-Face': 1.0,
            'Facenet': 1.0,
            'OpenFace': 0.8,
            'DeepFace': 1.0
        }
        
        total_confidence = 0
        total_weight = 0
        distances = []
        
        for result in results:
            weight = weights.get(result['model'], 1.0)
            total_confidence += result['confidence'] * weight
            total_weight += weight
            distances.append(result['distance'])
        
        avg_confidence = total_confidence / total_weight if total_weight > 0 else 0
        avg_distance = np.mean(distances)
        
        # Check if majority of models verified
        verified_count = sum(1 for r in results if r['verified'])
        is_verified = verified_count > len(results) / 2
        
        return {
            'distance': avg_distance,
            'confidence': avg_confidence,
            'verified': is_verified,
            'threshold': np.mean([r['threshold'] for r in results]),
            'num_models': len(results),
            'individual_results': results
        }
    
    def compare_faces(
        self,
        img1_path: str,
        img2_path: str
    ) -> Optional[Dict[str, Any]]:
        """Compare two faces with preprocessing and optional ensemble."""
        try:
            # Preprocess both images
            processed_img1, quality1 = self.preprocess_image(img1_path)
            processed_img2, quality2 = self.preprocess_image(img2_path)
            
            logger.debug(f"Image qualities: {quality1:.2f}, {quality2:.2f}")
            
            # Use ensemble or single model
            if self.use_ensemble:
                result = self.compare_faces_ensemble(processed_img1, processed_img2, quality1, quality2)
            else:
                result = self.compare_faces_single_model(
                    processed_img1, processed_img2, self.model_name, quality1, quality2
                )
            
            if result:
                result['quality1'] = quality1
                result['quality2'] = quality2
            
            return result
            
        except Exception as e:
            logger.debug(f"Face comparison failed: {str(e)[:100]}")
            return None
    
    def process_single_profile(
        self,
        profile: Dict[str, Any],
        target_image_path: str,
        image_field: str,
        index: int,
        total: int
    ) -> Optional[Dict[str, Any]]:
        """Process a single profile comparison."""
        if image_field not in profile or not profile[image_field]:
            return None
        
        image_url = profile[image_field]
        name = profile.get('name', 'Unknown')
        
        logger.info(f"\n[{index+1}/{total}] Processing: {name}")
        
        profile_img_path = self.download_and_cache_image(image_url)
        
        if not profile_img_path:
            logger.warning(f"  ⚠️  Could not load image")
            return None
        
        try:
            result = self.compare_faces(target_image_path, profile_img_path)
            
            if result:
                match_data = {
                    'profile': profile.copy(),
                    'distance': result['distance'],
                    'confidence': result['confidence'],
                    'verified': result['verified'],
                    'threshold': result['threshold'],
                    'quality1': result.get('quality1', 1.0),
                    'quality2': result.get('quality2', 1.0)
                }
                
                logger.info(f"  ✓ Distance: {result['distance']:.4f}, Confidence: {result['confidence']:.2%}")
                
                if self.use_ensemble:
                    logger.info(f"  📊 Ensemble: {result.get('num_models', 0)} models")
                
                return match_data
            
        except Exception as e:
            logger.warning(f"  ⚠️  Error: {str(e)[:100]}")
        
        return None
    
    def find_best_match(
        self,
        target_image_path: str,
        profiles_json_path: str,
        image_field: str = "imageUrl",
        min_confidence: float = 0.0,
        return_top_n: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching profile(s) with enhanced accuracy."""
        
        if not self.verify_image(target_image_path):
            logger.error("❌ Target image verification failed")
            return None
        
        logger.info(f"✓ Target image verified: {target_image_path}")
        
        if self.use_ensemble:
            logger.info(f"🔬 Using ensemble mode with {len(self.ensemble_models)} models: {self.ensemble_models}")
        
        profiles = self.load_linkedin_profiles(profiles_json_path)
        if not profiles:
            logger.error("❌ No profiles loaded")
            return None
        
        matches = []
        
        # Parallel or sequential processing
        if self.use_parallel:
            logger.info(f"⚡ Using parallel processing with {self.max_workers} workers")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_single_profile,
                        profile, target_image_path, image_field, i, len(profiles)
                    ): i for i, profile in enumerate(profiles)
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        matches.append(result)
        else:
            for i, profile in enumerate(profiles):
                result = self.process_single_profile(
                    profile, target_image_path, image_field, i, len(profiles)
                )
                if result:
                    matches.append(result)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processed: {len(matches)}/{len(profiles)} profiles")
        
        if not matches:
            logger.warning("❌ No valid comparisons completed")
            return None
        
        # Sort by confidence
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
            logger.info(f"    Quality: {match['quality1']:.2f} / {match['quality2']:.2f}")
        
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
    """
    Enhanced main function with maximum accuracy features.
    
    Args:
        json_data: File path or list of profiles
        target_img_data: File path, PIL Image, or numpy array
        **kwargs:
            - model_name: str (default: "VGG-Face")
            - use_ensemble: bool (default: False) - HIGHLY RECOMMENDED for accuracy
            - ensemble_models: List[str] - Custom model list
            - extract_faces: bool (default: True)
            - align_faces: bool (default: True)
            - enhance_images: bool (default: True) - NEW
            - detector_backend: str (default: "retinaface") - Better than opencv
            - face_quality_threshold: float (default: 0.0) - Filter low quality faces
            - min_confidence: float (default: 0.0)
            - use_parallel: bool (default: False) - Speed up processing
            - max_workers: int (default: 4)
    """
    
    # Extract kwargs
    model_name = kwargs.get('model_name', 'VGG-Face')
    use_ensemble = kwargs.get('use_ensemble', False)
    ensemble_models = kwargs.get('ensemble_models', None)
    enforce_detection = kwargs.get('enforce_detection', False)
    min_confidence = kwargs.get('min_confidence', 0.0)
    max_results = kwargs.get('max_results', 5)
    extract_faces = kwargs.get('extract_faces', True)
    align_faces = kwargs.get('align_faces', True)
    expand_face_region = kwargs.get('expand_face_region', 1.2)
    enhance_images = kwargs.get('enhance_images', True)
    detector_backend = kwargs.get('detector_backend', 'retinaface')
    face_quality_threshold = kwargs.get('face_quality_threshold', 0.0)
    use_parallel = kwargs.get('use_parallel', False)
    max_workers = kwargs.get('max_workers', 4)
    
    service = FaceRecognitionService(
        model_name=model_name,
        distance_metric="cosine",
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        cache_images=True,
        extract_faces=extract_faces,
        align_faces=align_faces,
        expand_face_region=expand_face_region,
        use_ensemble=use_ensemble,
        ensemble_models=ensemble_models,
        enhance_images=enhance_images,
        face_quality_threshold=face_quality_threshold,
        use_parallel=use_parallel,
        max_workers=max_workers
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

    # Run matching
    results = service.find_all_matches(
        target_image_path=target_image_path,
        profiles_json_path=profiles_json_path,
        image_field="imageUrl",
        max_results=max_results,
        min_confidence=min_confidence
    )

    return results




import json
from PIL import Image

profiles = json.load(open("./docs/sample_output.json"))
img = Image.open("./face_recognition_service/test_images/test_img6.jpg")

results = main(
    profiles, img,
    use_ensemble=True,           
    extract_faces=True,
    align_faces=True,
    enhance_images=True,           
    detector_backend='retinaface',  
    face_quality_threshold=0.3,     
    min_confidence=0.2
)


print(f'\n\n\nAAAAA NEW TRY FASTER\n\n\n')
'''
or use faster one but not that accurate:
'''
results1 = main(
    profiles, img,
    model_name="Facenet512",    
    extract_faces=True,
    enhance_images=True,
    use_parallel=True,            
    max_workers=4,
    min_confidence=0.3
)


