"""
Streamlit Demo App for Invoice Extraction System
Run with: streamlit run demo_app.py
"""

import streamlit as st
import json
import tempfile
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from executable import InvoiceExtractor
    import cv2
    import numpy as np
    from pdf2image import convert_from_path
except ImportError as e:
    st.error(f"Import error: {e}. Please install all dependencies.")
    st.stop()


# Page config
st.set_page_config(
    page_title="Invoice Field Extraction",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Invoice Field Extraction System")
st.markdown("""
This demo showcases an intelligent document AI system that extracts key fields from invoice documents.
Upload a PDF invoice to see the extraction in action!
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

use_api = st.sidebar.checkbox("Use Claude Vision API", value=True, 
                               help="Enable for higher accuracy (requires API key)")

api_key = None
if use_api:
    api_key = st.sidebar.text_input("Anthropic API Key", type="password",
                                     help="Enter your Anthropic API key")
    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            st.sidebar.success("✅ Using API key from environment")
        else:
            st.sidebar.warning("⚠️ API key required for VLM mode")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Expected Output Fields")
st.sidebar.markdown("""
- **Dealer Name** (fuzzy match)
- **Model Name** (exact match)
- **Horse Power** (numeric)
- **Asset Cost** (numeric)
- **Signature** (with bounding box)
- **Stamp** (with bounding box)
""")

# Main content
uploaded_file = st.file_uploader("Upload Invoice PDF", type=['pdf'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    st.success("✅ File uploaded successfully!")
    
    # Process button
    if st.button("🚀 Extract Fields", type="primary"):
        with st.spinner("Processing invoice..."):
            try:
                # Initialize extractor
                extractor = InvoiceExtractor(use_api=use_api, api_key=api_key)
                
                # Process document
                result = extractor.process_document(tmp_path)
                
                # Display results
                st.markdown("---")
                st.header("📋 Extraction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Extracted Fields")
                    
                    fields = result['fields']
                    
                    # Dealer Name
                    st.markdown(f"**Dealer Name:** {fields.get('dealer_name', 'N/A')}")
                    
                    # Model Name
                    st.markdown(f"**Model Name:** {fields.get('model_name', 'N/A')}")
                    
                    # Horse Power
                    hp = fields.get('horse_power', 0)
                    st.markdown(f"**Horse Power:** {hp} HP")
                    
                    # Asset Cost
                    cost = fields.get('asset_cost', 0)
                    st.markdown(f"**Asset Cost:** ₹{cost:,}")
                    
                    # Signature
                    sig = fields.get('signature', {})
                    sig_status = "✅ Present" if sig.get('present') else "❌ Not Found"
                    st.markdown(f"**Signature:** {sig_status}")
                    if sig.get('present'):
                        st.caption(f"Bbox: {sig.get('bbox')}")
                    
                    # Stamp
                    stamp = fields.get('stamp', {})
                    stamp_status = "✅ Present" if stamp.get('present') else "❌ Not Found"
                    st.markdown(f"**Stamp:** {stamp_status}")
                    if stamp.get('present'):
                        st.caption(f"Bbox: {stamp.get('bbox')}")
                
                with col2:
                    st.subheader("Metadata")
                    
                    # Confidence score
                    confidence = result.get('confidence', 0)
                    st.metric("Confidence Score", f"{confidence:.0%}")
                    
                    # Processing time
                    proc_time = result.get('processing_time_sec', 0)
                    st.metric("Processing Time", f"{proc_time:.2f}s")
                    
                    # Cost estimate
                    cost_est = result.get('cost_estimate_usd', 0)
                    st.metric("Cost Estimate", f"${cost_est:.4f}")
                    
                    # Document ID
                    st.caption(f"Document ID: {result.get('doc_id', 'N/A')}")
                
                # JSON output
                st.markdown("---")
                st.subheader("📝 JSON Output")
                st.json(result)
                
                # Download button
                json_str = json.dumps(result, indent=2)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_str,
                    file_name=f"{result.get('doc_id', 'result')}.json",
                    mime="application/json"
                )
                
                # Visualization
                st.markdown("---")
                st.subheader("🖼️ Document Preview")
                
                try:
                    # Convert PDF to image
                    images = convert_from_path(tmp_path, dpi=150)
                    if images:
                        img_array = np.array(images[0])
                        
                        # Draw bounding boxes
                        img_with_boxes = img_array.copy()
                        
                        # Draw signature box
                        if sig.get('present'):
                            bbox = sig['bbox']
                            cv2.rectangle(img_with_boxes, (bbox[0], bbox[1]), 
                                        (bbox[2], bbox[3]), (0, 255, 0), 5)
                            cv2.putText(img_with_boxes, 'Signature', 
                                      (bbox[0], bbox[1]-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Draw stamp box
                        if stamp.get('present'):
                            bbox = stamp['bbox']
                            cv2.rectangle(img_with_boxes, (bbox[0], bbox[1]), 
                                        (bbox[2], bbox[3]), (255, 0, 0), 5)
                            cv2.putText(img_with_boxes, 'Stamp', 
                                      (bbox[0], bbox[1]-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        
                        st.image(img_with_boxes, caption="Detected Elements", 
                               use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not generate preview: {e}")
                
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass

else:
    # Instructions
    st.info("👆 Upload a PDF invoice to get started")
    
    # Sample output
    st.markdown("---")
    st.subheader("📄 Sample Output")
    
    sample_output = {
        "doc_id": "invoice_001",
        "fields": {
            "dealer_name": "ABC Tractors Pvt Ltd",
            "model_name": "Mahindra 575 DI",
            "horse_power": 50,
            "asset_cost": 525000,
            "signature": {"present": True, "bbox": [1205, 1832, 1456, 1978]},
            "stamp": {"present": True, "bbox": [1150, 1650, 1350, 1850]}
        },
        "confidence": 0.96,
        "processing_time_sec": 3.8,
        "cost_estimate_usd": 0.003
    }
    
    st.json(sample_output)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built for IDFC GenAI Hackathon - Convolve 4.0</p>
    <p>Powered by Claude Vision API + PaddleOCR</p>
</div>
""", unsafe_allow_html=True)
