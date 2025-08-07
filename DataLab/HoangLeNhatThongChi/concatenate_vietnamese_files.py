#!/usr/bin/env python3
"""
Script to process Vietnamese text files by:
1. Removing the 4th line from each file
2. Concatenating all remaining lines into one paragraph separated by spaces
"""

import os
import re
from pathlib import Path

def process_file_content(content):
    """
    Keep first 3 lines as they are, remove the 4th line, and concatenate lines 5+ into one paragraph.
    
    Args:
        content (str): Original file content
        
    Returns:
        str: Processed content with first 3 lines preserved, 4th line removed, and remaining lines concatenated
    """
    lines = content.split('\n')
    
    if len(lines) <= 3:
        # If file has 3 or fewer lines, return as is
        return content
    
    # Keep the first 3 lines
    first_three_lines = lines[:3]
    
    # Remove the 4th line and get lines from 5th onwards (index 4+)
    remaining_lines = lines[4:] if len(lines) > 4 else []
    
    # Filter out empty lines and strip whitespace from remaining lines
    non_empty_remaining = [line.strip() for line in remaining_lines if line.strip()]
    
    # Create the result
    result_lines = first_three_lines[:]
    
    # If there are remaining lines, concatenate them into one paragraph
    if non_empty_remaining:
        concatenated_paragraph = ' '.join(non_empty_remaining)
        result_lines.append(concatenated_paragraph)
    
    return '\n'.join(result_lines)

def process_vietnamese_files():
    """
    Process all Vietnamese text files to remove 4th line and concatenate.
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
            
            # Process the content
            processed_content = process_file_content(original_content)
            
            # Write back the processed content
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            processed_count += 1
            print(f"  ✓ Processed {txt_file.name}")
            
            # Show a preview of the result for the first few files
            if processed_count <= 3:
                preview = processed_content[:100] + "..." if len(processed_content) > 100 else processed_content
                print(f"    Preview: {preview}")
            
        except Exception as e:
            print(f"  ✗ Error processing {txt_file.name}: {str(e)}")
    
    print(f"\nProcessing complete! Successfully processed {processed_count}/{len(txt_files)} files.")

def main():
    """Main function"""
    print("Processing Vietnamese text files...")
    print("- Keeping the first 3 lines as they are")
    print("- Removing the 4th line from each file")
    print("- Concatenating lines 5+ into one paragraph with spaces")
    print("-" * 60)
    
    process_vietnamese_files()

if __name__ == "__main__":
    main() 