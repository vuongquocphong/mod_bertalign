#!/usr/bin/env python3
"""
Script to split the first line of Chinese text files into 3 separate lines.
Format: "第X回 [title part 1] [title part 2]" becomes:
Line 1: "第X回"
Line 2: "[title part 1]"  
Line 3: "[title part 2]"
"""

import os
import re
from pathlib import Path

def split_first_line(content):
    """
    Split the first line of Chinese text into 3 lines.
    
    Args:
        content (str): Full file content
        
    Returns:
        str: Modified content with first line split into 3 lines
    """
    lines = content.split('\n')
    if not lines:
        return content
    
    first_line = lines[0]
    
    # Pattern to match: 第X回 [title part 1] [title part 2]
    # The title parts are separated by spaces
    match = re.match(r'^(第\w+回)\s+(.+)$', first_line)
    
    if not match:
        # If pattern doesn't match, return original content
        return content
    
    chapter_part = match.group(1)  # 第X回
    title_part = match.group(2)    # remaining title
    
    # Split the title part - look for natural break points
    # Many titles have two parts separated by space, like "郑宣妃宠冠后宫 王世子废居幽室"
    title_parts = title_part.split(' ', 1)  # Split on first space only
    
    if len(title_parts) == 2:
        title_part1 = title_parts[0]
        title_part2 = title_parts[1]
    else:
        # If no clear split, try to split in the middle by characters
        title_chars = list(title_part.replace(' ', ''))
        if len(title_chars) >= 4:
            mid_point = len(title_chars) // 2
            title_part1 = ''.join(title_chars[:mid_point])
            title_part2 = ''.join(title_chars[mid_point:])
        else:
            title_part1 = ""
            title_part2 = title_part
    
    # Create the new first 3 lines
    new_lines = [chapter_part]
    if title_part1:
        new_lines.append(title_part1)
    if title_part2:
        new_lines.append(title_part2)
    
    # Combine with the rest of the content
    remaining_lines = lines[1:]
    all_lines = new_lines + remaining_lines
    
    return '\n'.join(all_lines)

def process_chinese_files():
    """
    Process all Chinese text files to split their first lines.
    """
    chinese_dir = Path("chinese")
    
    if not chinese_dir.exists():
        print(f"Directory {chinese_dir} does not exist!")
        return
    
    # Get all .txt files in the chinese directory
    txt_files = list(chinese_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {chinese_dir}")
        return
    
    print(f"Found {len(txt_files)} .txt files to process:")
    
    processed_count = 0
    
    for txt_file in sorted(txt_files):
        try:
            print(f"Processing: {txt_file.name}")
            
            # Read the original file
            with open(txt_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Split the first line
            modified_content = split_first_line(original_content)
            
            # Write back the modified content
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            processed_count += 1
            print(f"  ✓ Split header in {txt_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error processing {txt_file.name}: {str(e)}")
    
    print(f"\nProcessing complete! Successfully processed {processed_count}/{len(txt_files)} files.")

def main():
    """Main function"""
    print("Splitting Chinese file headers...")
    print("Converting first line format from '第X回 title1 title2' to 3 separate lines")
    print("-" * 60)
    
    process_chinese_files()

if __name__ == "__main__":
    main() 