"""
IDFC GenAI Hackathon - Invoice Field Extraction System
Main executable for extracting fields from invoice documents
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import base64

# Vision and OCR imports
try:
    from pdf2image import convert_from_path
    from PIL import Image
    import cv2
    import numpy as np
    from paddleocr import PaddleOCR
except ImportError:
    print("Warning: Some OCR dependencies not installed")

# API imports for vision models
import anthropic
import re
from difflib import SequenceMatcher


class InvoiceExtractor:
    """Main class for invoice field extraction"""
    
    def __init__(self, use_api=True, api_key=None):
        self.use_api = use_api
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        # Initialize OCR
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', 
                                use_gpu=False, show_log=False)
        except:
            self.ocr = None
            print("PaddleOCR not available, using API-only mode")
        
        # Load master data
        self.dealer_master = self._load_dealer_master()
        self.model_master = self._load_model_master()
        
    def _load_dealer_master(self) -> List[str]:
        """Load dealer master list for fuzzy matching"""
        # In production, load from CSV/database
        return [
            "ABC Tractors Pvt Ltd",
            "Shree Krishna Auto",
            "Mahindra Dealership",
            "John Deere India",
            "TAFE Motors",
            "Escorts Tractors"
        ]
    
    def _load_model_master(self) -> List[str]:
        """Load model master list for exact matching"""
        return [
            "Mahindra 575 DI",
            "John Deere 5050D",
            "Swaraj 855 FE",
            "New Holland 3630",
            "Massey Ferguson 7250"
        ]
    
    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """Convert PDF to images"""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            return [np.array(img) for img in images]
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []
    
    def extract_with_paddle_ocr(self, image: np.ndarray) -> str:
        """Extract text using PaddleOCR"""
        if self.ocr is None:
            return ""
        
        result = self.ocr.ocr(image, cls=True)
        text_lines = []
        
        for line in result:
            if line:
                for word_info in line:
                    text_lines.append(word_info[1][0])
        
        return "\n".join(text_lines)
    
    def extract_with_claude_vision(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract fields using Claude Vision API"""
        if not self.api_key:
            return {}
        
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            
            prompt = """Analyze this invoice/quotation document and extract the following fields:

1. Dealer Name (company/shop name)
2. Model Name (tractor or vehicle model)
3. Horse Power (numeric value only, e.g., if "50 HP" then return 50)
4. Asset Cost (total cost in digits only, no currency symbols)
5. Signature presence (true/false and bounding box if present)
6. Stamp presence (true/false and bounding box if present)

Return ONLY a JSON object in this exact format:
{
  "dealer_name": "extracted dealer name",
  "model_name": "extracted model name",
  "horse_power": numeric_value,
  "asset_cost": numeric_value,
  "signature": {"present": true/false, "bbox": [x1, y1, x2, y2]},
  "stamp": {"present": true/false, "bbox": [x1, y1, x2, y2]}
}

Bounding box coordinates should be [left, top, right, bottom] in pixels.
If a field is not found, use null for text fields and 0 for numeric fields."""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            response_text = message.content[0].text
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Claude API error: {e}")
        
        return {}
    
    def fuzzy_match_dealer(self, extracted: str) -> Tuple[str, float]:
        """Fuzzy match dealer name against master list"""
        if not extracted:
            return "", 0.0
        
        best_match = ""
        best_score = 0.0
        
        for dealer in self.dealer_master:
            score = SequenceMatcher(None, extracted.lower(), dealer.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = dealer
        
        return best_match, best_score
    
    def exact_match_model(self, extracted: str) -> Tuple[str, bool]:
        """Exact match model name against master list"""
        if not extracted:
            return "", False
        
        for model in self.model_master:
            if extracted.lower().strip() == model.lower().strip():
                return model, True
        
        return extracted, False
    
    def detect_signature_stamp(self, image: np.ndarray) -> Tuple[Dict, Dict]:
        """Detect signature and stamp using CV techniques"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple contour-based detection
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        signature = {"present": False, "bbox": [0, 0, 0, 0]}
        stamp = {"present": False, "bbox": [0, 0, 0, 0]}
        
        # Heuristic: signatures are usually smaller, stamps are circular/square
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1000 < area < 50000:  # Adjust based on image size
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                
                # Stamp detection (more square-like)
                if 0.8 < aspect_ratio < 1.2 and area > 5000:
                    stamp = {"present": True, "bbox": [x, y, x+w, y+h]}
                # Signature detection (more elongated)
                elif area > 2000:
                    signature = {"present": True, "bbox": [x, y, x+w, y+h]}
        
        return signature, stamp
    
    def calculate_confidence(self, fields: Dict) -> float:
        """Calculate overall confidence score"""
        scores = []
        
        # Dealer name confidence
        if fields.get('dealer_name'):
            _, dealer_score = self.fuzzy_match_dealer(fields['dealer_name'])
            scores.append(dealer_score)
        
        # Model name confidence
        if fields.get('model_name'):
            _, model_match = self.exact_match_model(fields['model_name'])
            scores.append(1.0 if model_match else 0.5)
        
        # Numeric fields confidence
        if fields.get('horse_power', 0) > 0:
            scores.append(0.95)
        if fields.get('asset_cost', 0) > 0:
            scores.append(0.95)
        
        # Signature/stamp confidence
        if fields.get('signature', {}).get('present'):
            scores.append(0.9)
        if fields.get('stamp', {}).get('present'):
            scores.append(0.9)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """Main processing pipeline for a single document"""
        start_time = time.time()
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        if not images:
            return self._empty_result(pdf_path)
        
        # Process first page (most invoices are single page)
        image = images[0]
        
        # Extract fields using API
        if self.use_api and self.api_key:
            fields = self.extract_with_claude_vision(image)
        else:
            # Fallback to OCR + rule-based extraction
            ocr_text = self.extract_with_paddle_ocr(image)
            fields = self._rule_based_extraction(ocr_text, image)
        
        # Post-process and validate
        fields = self._post_process_fields(fields)
        
        # Calculate confidence
        confidence = self.calculate_confidence(fields)
        
        # Calculate cost estimate
        cost_estimate = 0.003 if self.use_api else 0.0
        
        processing_time = time.time() - start_time
        
        doc_id = Path(pdf_path).stem
        
        return {
            "doc_id": doc_id,
            "fields": fields,
            "confidence": round(confidence, 2),
            "processing_time_sec": round(processing_time, 2),
            "cost_estimate_usd": cost_estimate
        }
    
    def _rule_based_extraction(self, text: str, image: np.ndarray) -> Dict[str, Any]:
        """Fallback rule-based extraction from OCR text"""
        fields = {
            "dealer_name": "",
            "model_name": "",
            "horse_power": 0,
            "asset_cost": 0,
            "signature": {"present": False, "bbox": [0, 0, 0, 0]},
            "stamp": {"present": False, "bbox": [0, 0, 0, 0]}
        }
        
        lines = text.split('\n')
        
        # Extract dealer name (usually at top)
        for line in lines[:5]:
            if len(line) > 5 and any(keyword in line.lower() for keyword in ['pvt', 'ltd', 'dealer', 'auto', 'motors']):
                fields['dealer_name'] = line.strip()
                break
        
        # Extract model name
        for model in self.model_master:
            if model.lower() in text.lower():
                fields['model_name'] = model
                break
        
        # Extract horse power
        hp_match = re.search(r'(\d+)\s*hp', text, re.IGNORECASE)
        if hp_match:
            fields['horse_power'] = int(hp_match.group(1))
        
        # Extract cost
        cost_matches = re.findall(r'(?:rs\.?|₹)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
        if cost_matches:
            cost_str = cost_matches[-1].replace(',', '')
            fields['asset_cost'] = int(float(cost_str))
        
        # Detect signature and stamp
        signature, stamp = self.detect_signature_stamp(image)
        fields['signature'] = signature
        fields['stamp'] = stamp
        
        return fields
    
    def _post_process_fields(self, fields: Dict) -> Dict[str, Any]:
        """Post-process and validate extracted fields"""
        # Fuzzy match dealer name
        if fields.get('dealer_name'):
            matched_dealer, score = self.fuzzy_match_dealer(fields['dealer_name'])
            if score >= 0.7:
                fields['dealer_name'] = matched_dealer
        
        # Exact match model name
        if fields.get('model_name'):
            matched_model, _ = self.exact_match_model(fields['model_name'])
            fields['model_name'] = matched_model
        
        return fields
    
    def _empty_result(self, pdf_path: str) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "doc_id": Path(pdf_path).stem,
            "fields": {
                "dealer_name": "",
                "model_name": "",
                "horse_power": 0,
                "asset_cost": 0,
                "signature": {"present": False, "bbox": [0, 0, 0, 0]},
                "stamp": {"present": False, "bbox": [0, 0, 0, 0]}
            },
            "confidence": 0.0,
            "processing_time_sec": 0.0,
            "cost_estimate_usd": 0.0
        }


def main():
    parser = argparse.ArgumentParser(description='Invoice Field Extraction')
    parser.add_argument('--input', type=str, required=True, help='Input PDF file or directory')
    parser.add_argument('--output', type=str, default='output.json', help='Output JSON file')
    parser.add_argument('--use-api', action='store_true', help='Use Claude Vision API')
    parser.add_argument('--api-key', type=str, help='Anthropic API key')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = InvoiceExtractor(use_api=args.use_api, api_key=args.api_key)
    
    # Process single file or directory
    input_path = Path(args.input)
    results = []
    
    if input_path.is_file():
        result = extractor.process_document(str(input_path))
        results.append(result)
    elif input_path.is_dir():
        pdf_files = list(input_path.glob('*.pdf'))
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            result = extractor.process_document(str(pdf_file))
            results.append(result)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed {len(results)} documents")
    print(f"Results saved to {args.output}")
    
    # Print summary statistics
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    avg_time = sum(r['processing_time_sec'] for r in results) / len(results) if results else 0
    total_cost = sum(r['cost_estimate_usd'] for r in results)
    
    print(f"\nSummary:")
    print(f"Average Confidence: {avg_confidence:.2%}")
    print(f"Average Processing Time: {avg_time:.2f}s")
    print(f"Total Cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
