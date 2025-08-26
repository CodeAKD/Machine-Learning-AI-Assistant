#!/usr/bin/env python3
"""
Test script to verify PDF path finding functionality
"""

import os
import json
from streamlit_app import find_original_pdf_path

def test_pdf_paths():
    """Test PDF path finding for metadata files"""
    
    # Load metadata to get filenames
    metadata_file = "embeddings/metadata.json"
    if not os.path.exists(metadata_file):
        print("Metadata file not found!")
        return
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print("Testing PDF path finding for all documents:")
    print("=" * 60)
    
    found_count = 0
    total_count = len(metadata)
    
    for doc in metadata:
        filename = doc['filename']
        pdf_path = find_original_pdf_path(filename)
        
        if pdf_path and os.path.exists(pdf_path):
            print(f"✅ FOUND: {filename}")
            print(f"   Path: {pdf_path}")
            found_count += 1
        else:
            print(f"❌ NOT FOUND: {filename}")
        print()
    
    print("=" * 60)
    print(f"Summary: {found_count}/{total_count} PDFs found ({found_count/total_count*100:.1f}%)")
    
    if found_count < total_count:
        print("\nChecking ML Books directory contents:")
        ml_books_dir = "ML Books"
        if os.path.exists(ml_books_dir):
            pdf_files = [f for f in os.listdir(ml_books_dir) if f.endswith('.pdf')]
            print(f"Found {len(pdf_files)} PDF files in ML Books:")
            for pdf in pdf_files[:10]:  # Show first 10
                print(f"  - {pdf}")
            if len(pdf_files) > 10:
                print(f"  ... and {len(pdf_files) - 10} more")
        else:
            print("ML Books directory not found!")

if __name__ == "__main__":
    test_pdf_paths()