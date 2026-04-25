"""
Evaluation script for invoice extraction system
Computes DLA and field-level metrics against ground truth
"""

import json
import argparse
from typing import Dict, List
from pathlib import Path
from difflib import SequenceMatcher


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def fuzzy_match(s1: str, s2: str, threshold: float = 0.9) -> bool:
    """Check if two strings match with fuzzy threshold"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio() >= threshold


def exact_match(s1: str, s2: str) -> bool:
    """Check exact string match (case-insensitive)"""
    return s1.lower().strip() == s2.lower().strip()


def numeric_match(n1: float, n2: float, tolerance: float = 0.05) -> bool:
    """Check if two numbers match within tolerance"""
    if n2 == 0:
        return n1 == 0
    return abs(n1 - n2) / n2 <= tolerance


def evaluate_single_document(pred: Dict, gt: Dict) -> Dict:
    """
    Evaluate single document prediction against ground truth
    
    Args:
        pred: Predicted fields
        gt: Ground truth fields
    
    Returns:
        Evaluation results with field-level scores
    """
    results = {
        'doc_id': pred.get('doc_id', ''),
        'fields': {},
        'all_correct': True
    }
    
    pred_fields = pred.get('fields', {})
    gt_fields = gt.get('fields', {})
    
    # Evaluate Dealer Name (fuzzy match ≥90%)
    dealer_correct = fuzzy_match(
        pred_fields.get('dealer_name', ''),
        gt_fields.get('dealer_name', ''),
        threshold=0.9
    )
    results['fields']['dealer_name'] = dealer_correct
    
    # Evaluate Model Name (exact match)
    model_correct = exact_match(
        pred_fields.get('model_name', ''),
        gt_fields.get('model_name', '')
    )
    results['fields']['model_name'] = model_correct
    
    # Evaluate Horse Power (exact match)
    hp_correct = pred_fields.get('horse_power', 0) == gt_fields.get('horse_power', 0)
    results['fields']['horse_power'] = hp_correct
    
    # Evaluate Asset Cost (exact match)
    cost_correct = pred_fields.get('asset_cost', 0) == gt_fields.get('asset_cost', 0)
    results['fields']['asset_cost'] = cost_correct
    
    # Evaluate Signature (presence + IoU ≥ 0.5)
    pred_sig = pred_fields.get('signature', {})
    gt_sig = gt_fields.get('signature', {})
    
    sig_present_match = pred_sig.get('present') == gt_sig.get('present')
    sig_iou = 0.0
    
    if pred_sig.get('present') and gt_sig.get('present'):
        sig_iou = calculate_iou(pred_sig.get('bbox', [0,0,0,0]), 
                               gt_sig.get('bbox', [0,0,0,0]))
        sig_correct = sig_present_match and sig_iou >= 0.5
    else:
        sig_correct = sig_present_match
    
    results['fields']['signature'] = sig_correct
    results['signature_iou'] = sig_iou
    
    # Evaluate Stamp (presence + IoU ≥ 0.5)
    pred_stamp = pred_fields.get('stamp', {})
    gt_stamp = gt_fields.get('stamp', {})
    
    stamp_present_match = pred_stamp.get('present') == gt_stamp.get('present')
    stamp_iou = 0.0
    
    if pred_stamp.get('present') and gt_stamp.get('present'):
        stamp_iou = calculate_iou(pred_stamp.get('bbox', [0,0,0,0]), 
                                 gt_stamp.get('bbox', [0,0,0,0]))
        stamp_correct = stamp_present_match and stamp_iou >= 0.5
    else:
        stamp_correct = stamp_present_match
    
    results['fields']['stamp'] = stamp_correct
    results['stamp_iou'] = stamp_iou
    
    # Document-level correctness
    results['all_correct'] = all(results['fields'].values())
    
    return results


def evaluate_batch(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    Evaluate batch of predictions
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
    
    Returns:
        Comprehensive evaluation metrics
    """
    # Create lookup dictionary for ground truth
    gt_lookup = {gt['doc_id']: gt for gt in ground_truth}
    
    # Evaluate each prediction
    doc_results = []
    for pred in predictions:
        doc_id = pred.get('doc_id', '')
        if doc_id in gt_lookup:
            result = evaluate_single_document(pred, gt_lookup[doc_id])
            doc_results.append(result)
    
    # Calculate aggregate metrics
    total_docs = len(doc_results)
    
    if total_docs == 0:
        return {'error': 'No matching documents found'}
    
    # Document-Level Accuracy (DLA)
    dla = sum(1 for r in doc_results if r['all_correct']) / total_docs
    
    # Field-level accuracy
    field_accuracy = {}
    for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost', 'signature', 'stamp']:
        correct = sum(1 for r in doc_results if r['fields'].get(field, False))
        field_accuracy[field] = correct / total_docs
    
    # IoU statistics for signature and stamp
    sig_ious = [r['signature_iou'] for r in doc_results if r['signature_iou'] > 0]
    stamp_ious = [r['stamp_iou'] for r in doc_results if r['stamp_iou'] > 0]
    
    metrics = {
        'document_level_accuracy': dla,
        'field_level_accuracy': field_accuracy,
        'total_documents': total_docs,
        'correct_documents': sum(1 for r in doc_results if r['all_correct']),
        'signature_mean_iou': sum(sig_ious) / len(sig_ious) if sig_ious else 0.0,
        'stamp_mean_iou': sum(stamp_ious) / len(stamp_ious) if stamp_ious else 0.0,
        'detailed_results': doc_results
    }
    
    return metrics


def print_evaluation_report(metrics: Dict):
    """Print formatted evaluation report"""
    print("=" * 60)
    print("INVOICE EXTRACTION EVALUATION REPORT")
    print("=" * 60)
    print()
    
    print(f"Total Documents Evaluated: {metrics['total_documents']}")
    print(f"Correct Documents: {metrics['correct_documents']}")
    print()
    
    print(f"📊 DOCUMENT-LEVEL ACCURACY (DLA): {metrics['document_level_accuracy']:.2%}")
    print(f"   Target: ≥95% | Status: {'✅ PASS' if metrics['document_level_accuracy'] >= 0.95 else '❌ FAIL'}")
    print()
    
    print("📋 FIELD-LEVEL ACCURACY:")
    for field, accuracy in metrics['field_level_accuracy'].items():
        print(f"   {field:20s}: {accuracy:.2%}")
    print()
    
    print("🎯 BOUNDING BOX METRICS:")
    print(f"   Signature Mean IoU: {metrics['signature_mean_iou']:.3f}")
    print(f"   Stamp Mean IoU:     {metrics['stamp_mean_iou']:.3f}")
    print()
    
    # Error analysis
    failed_docs = [r for r in metrics['detailed_results'] if not r['all_correct']]
    if failed_docs:
        print("❌ FAILED DOCUMENTS:")
        for doc in failed_docs[:10]:  # Show first 10
            print(f"   {doc['doc_id']}:")
            for field, correct in doc['fields'].items():
                if not correct:
                    print(f"      - {field} ❌")
        print()
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate invoice extraction results')
    parser.add_argument('--predictions', type=str, required=True, 
                       help='Path to predictions JSON')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth JSON')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Load predictions and ground truth
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    
    # Ensure both are lists
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    
    # Evaluate
    metrics = evaluate_batch(predictions, ground_truth)
    
    # Print report
    print_evaluation_report(metrics)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
