"""
Utility functions for invoice processing pipeline
"""

import re
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union for two bounding boxes
    
    Args:
        box1, box2: Bounding boxes as [x1, y1, x2, y2]
    
    Returns:
        IoU score (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def fuzzy_string_match(s1: str, s2: str) -> float:
    """
    Calculate fuzzy string similarity using sequence matching
    
    Args:
        s1, s2: Strings to compare
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    return SequenceMatcher(None, s1.lower().strip(), s2.lower().strip()).ratio()


def extract_numeric_value(text: str, pattern: str = r'\d+') -> Optional[int]:
    """
    Extract numeric value from text using regex
    
    Args:
        text: Input text
        pattern: Regex pattern for extraction
    
    Returns:
        Extracted integer value or None
    """
    match = re.search(pattern, text.replace(',', ''))
    return int(match.group()) if match else None


def normalize_dealer_name(name: str) -> str:
    """
    Normalize dealer name for consistent matching
    
    Args:
        name: Raw dealer name
    
    Returns:
        Normalized name
    """
    # Remove common suffixes
    suffixes = ['pvt ltd', 'private limited', 'ltd', 'inc', 'llp', 'llc']
    normalized = name.lower().strip()
    
    for suffix in suffixes:
        normalized = normalized.replace(suffix, '').strip()
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized.title()


def detect_text_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect text regions in image using MSER
    
    Args:
        image: Input image
    
    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Create MSER detector
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    bboxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        # Filter small regions
        if w > 20 and h > 10:
            bboxes.append((x, y, w, h))
    
    return bboxes


def validate_extraction(fields: Dict) -> Tuple[bool, List[str]]:
    """
    Validate extracted fields for consistency
    
    Args:
        fields: Extracted field dictionary
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check dealer name
    if not fields.get('dealer_name'):
        errors.append("Missing dealer name")
    
    # Check model name
    if not fields.get('model_name'):
        errors.append("Missing model name")
    
    # Validate horse power range
    hp = fields.get('horse_power', 0)
    if hp < 10 or hp > 150:
        errors.append(f"Horse power {hp} out of valid range (10-150)")
    
    # Validate asset cost
    cost = fields.get('asset_cost', 0)
    if cost < 100000 or cost > 10000000:
        errors.append(f"Asset cost {cost} out of valid range (100K-10M)")
    
    # Check signature/stamp structure
    for field_name in ['signature', 'stamp']:
        field = fields.get(field_name, {})
        if not isinstance(field, dict) or 'present' not in field or 'bbox' not in field:
            errors.append(f"Invalid {field_name} structure")
        elif field['present'] and len(field['bbox']) != 4:
            errors.append(f"Invalid {field_name} bounding box")
    
    return len(errors) == 0, errors


def parse_indian_currency(text: str) -> Optional[int]:
    """
    Parse Indian currency notation (lakhs, crores)
    
    Args:
        text: Currency text (e.g., "5.25 lakh", "₹525000")
    
    Returns:
        Parsed amount as integer
    """
    text = text.lower().strip()
    
    # Remove currency symbols
    text = re.sub(r'[₹$,]', '', text)
    
    # Handle lakhs and crores
    if 'crore' in text or 'cr' in text:
        match = re.search(r'([\d.]+)\s*(?:crore|cr)', text)
        if match:
            return int(float(match.group(1)) * 10000000)
    
    if 'lakh' in text or 'lac' in text:
        match = re.search(r'([\d.]+)\s*(?:lakh|lac)', text)
        if match:
            return int(float(match.group(1)) * 100000)
    
    # Direct numeric extraction
    match = re.search(r'\d+', text.replace(',', ''))
    return int(match.group()) if match else None


def visualize_extractions(image: np.ndarray, fields: Dict, output_path: str):
    """
    Visualize extracted bounding boxes on image
    
    Args:
        image: Input image
        fields: Extracted fields with bounding boxes
        output_path: Path to save visualization
    """
    vis_image = image.copy()
    
    # Draw signature box
    if fields.get('signature', {}).get('present'):
        bbox = fields['signature']['bbox']
        cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                     (0, 255, 0), 3)
        cv2.putText(vis_image, 'Signature', (bbox[0], bbox[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Draw stamp box
    if fields.get('stamp', {}).get('present'):
        bbox = fields['stamp']['bbox']
        cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                     (255, 0, 0), 3)
        cv2.putText(vis_image, 'Stamp', (bbox[0], bbox[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, vis_image)


def batch_process_stats(results: List[Dict]) -> Dict[str, any]:
    """
    Calculate statistics for batch processing results
    
    Args:
        results: List of processing results
    
    Returns:
        Statistics dictionary
    """
    total = len(results)
    
    if total == 0:
        return {}
    
    stats = {
        'total_documents': total,
        'avg_confidence': sum(r['confidence'] for r in results) / total,
        'avg_processing_time': sum(r['processing_time_sec'] for r in results) / total,
        'total_cost': sum(r['cost_estimate_usd'] for r in results),
        'high_confidence_count': sum(1 for r in results if r['confidence'] >= 0.9),
        'low_confidence_count': sum(1 for r in results if r['confidence'] < 0.7),
    }
    
    # Field-level statistics
    field_stats = {}
    for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
        extracted = sum(1 for r in results if r['fields'].get(field))
        field_stats[field] = {
            'extracted_count': extracted,
            'extraction_rate': extracted / total
        }
    
    stats['field_statistics'] = field_stats
    
    return stats


def create_master_lists_from_data(ocr_texts: List[str]) -> Tuple[List[str], List[str]]:
    """
    Extract potential dealer and model names from OCR texts
    (Useful for creating master lists from unlabeled data)
    
    Args:
        ocr_texts: List of OCR extracted texts
    
    Returns:
        (dealer_candidates, model_candidates)
    """
    dealers = set()
    models = set()
    
    for text in ocr_texts:
        lines = text.split('\n')
        
        # Look for dealer patterns (containing company keywords)
        for line in lines[:10]:  # Usually in first few lines
            if any(kw in line.lower() for kw in ['pvt', 'ltd', 'dealer', 'motors', 'auto']):
                if 10 < len(line) < 100:
                    dealers.add(line.strip())
        
        # Look for model patterns (brand + numbers)
        for line in lines:
            if re.search(r'(mahindra|john deere|swaraj|massey|new holland)\s+\d+', 
                        line, re.IGNORECASE):
                if 5 < len(line) < 50:
                    models.add(line.strip())
    
    return sorted(dealers), sorted(models)
