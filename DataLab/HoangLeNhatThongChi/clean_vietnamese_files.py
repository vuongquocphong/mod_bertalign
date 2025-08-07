#!/usr/bin/env python3
"""
Script to clean Vietnamese text files by removing page numbers and URLs.
Processes all .txt files in the vietnamese directory.
"""

import os
import re
from pathlib import Path

def clean_text(text):
    """
    Remove page numbers and URLs from text content.
    
    Args:
        text (str): Input text content
        
    Returns:
        str: Cleaned text content
    """
    # Remove page numbers (e.g., "Page 4", "Page 10", etc.)
    text = re.sub(r'Page\s+\d+\s*', '', text, flags=re.IGNORECASE)
    
    # Remove URLs (http and https)
    text = re.sub(r'https?://[^\s\n]+', '', text)
    
    # Remove any standalone "http://www.lichsuvietnam.info" lines
    text = re.sub(r'^\s*http://www\.lichsuvietnam\.info\s*$', '', text, flags=re.MULTILINE)
    
    # Remove the header line "Hoàng Lê Nhất Thống Chí – Ngô gia văn phái"
    text = re.sub(r'^\s*Hoàng Lê Nhất Thống Chí – Ngô gia văn phái\s*$', '', text, flags=re.MULTILINE)
    
    # Fix Vietnamese character encoding issues
    text = text.replace('ñ', 'đ')  # Replace lowercase ñ with đ
    text = text.replace('ð', 'Đ')  # Replace uppercase ð with Đ
    
    # Remove multiple consecutive empty lines and replace with single empty line
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text

def process_vietnamese_files():
    """
    Process all .txt files in the vietnamese directory to remove page numbers and URLs.
    """
    vietnamese_dir = Path("vietnamese")
    
    if not vietnamese_dir.exists():
        print(f"Directory {vietnamese_dir} does not exist!")
        return
    
    # Get all .txt files in the vietnamese directory
    txt_files = list(vietnamese_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {vietnamese_dir}")
        return
    
    print(f"Found {len(txt_files)} .txt files to process:")
    
    processed_count = 0
    
    for txt_file in sorted(txt_files):
        try:
            print(f"Processing: {txt_file.name}")
            
            # Read the original file
            with open(txt_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Clean the content
            cleaned_content = clean_text(original_content)
            
            # Write back the cleaned content
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            processed_count += 1
            print(f"  ✓ Cleaned {txt_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error processing {txt_file.name}: {str(e)}")
    
    print(f"\nProcessing complete! Successfully processed {processed_count}/{len(txt_files)} files.")

def main():
    """Main function"""
    print("Cleaning Vietnamese text files...")
    print("Removing page numbers, URLs, and fixing character encoding from all .txt files in the vietnamese directory.")
    print("- Removing page numbers (Page <num>)")
    print("- Removing URLs (http/https)")
    print("- Fixing character encoding: ñ → đ, ð → Đ")
    print("-" * 60)
    
    process_vietnamese_files()

if __name__ == "__main__":
    main() 