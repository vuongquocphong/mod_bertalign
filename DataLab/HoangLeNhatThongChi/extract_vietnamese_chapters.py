#!/usr/bin/env python3
"""
Script to extract chapters from HoangLeNhatThongChi Vietnamese PDF file.
Skips the first two pages and extracts each chapter to separate text files.
"""

import re
import os
from pathlib import Path

def vietnamese_ordinal_to_number(ordinal):
    """Convert Vietnamese ordinal words to numbers"""
    ordinal_map = {
        'nhất': '01', 'hai': '02', 'ba': '03', 'bốn': '04', 'năm': '05',
        'sáu': '06', 'bảy': '07', 'tám': '08', 'chín': '09', 'mười': '10',
        'mười một': '11', 'mười hai': '12', 'mười ba': '13', 'mười bốn': '14', 'mười lăm': '15',
        'mười sáu': '16', 'mười bảy': '17', 'mười tám': '18', 'mười chín': '19', 'hai mười': '20',
        'hai mươi': '20', 'hai mười một': '21', 'hai mươi một': '21', 'hai mười hai': '22', 'hai mươi hai': '22',
        'hai mười ba': '23', 'hai mươi ba': '23', 'hai mười bốn': '24', 'hai mươi bốn': '24',
        'hai mười lăm': '25', 'hai mươi lăm': '25'
    }
    
    # Clean up the ordinal text
    ordinal = ordinal.strip().lower()
    
    # Try direct mapping first
    if ordinal in ordinal_map:
        return ordinal_map[ordinal]
    
    # If not found, try to extract number and return formatted
    # This is a fallback for any ordinals not in our map
    for i in range(1, 100):
        if str(i) in ordinal:
            return f"{i:02d}"
    
    return "00"  # Default if not found

def extract_chapters_with_pymupdf():
    """Extract chapters using PyMuPDF (fitz) library"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not available. Please install with: pip install PyMuPDF")
        return False
    
    pdf_path = "viet_text.pdf"
    output_dir = "vietnamese"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    print(f"PDF has {len(doc)} pages")
    
    # Extract text from page 3 onwards (skip first 2 pages)
    full_text = ""
    for page_num in range(2, len(doc)):  # Start from page 3 (index 2)
        page = doc[page_num]
        text = page.get_text()
        full_text += text + "\n"
    
    doc.close()
    
    # Split into chapters based on chapter markers
    # Look for patterns like "Hồi thứ nhất", "Hồi thứ hai", etc.
    chapter_pattern = r'Hồi thứ ([^.\n]*?)(?:\.|:|\n)'
    chapter_matches = re.finditer(chapter_pattern, full_text)
    
    chapters = []
    last_end = 0
    
    for match in chapter_matches:
        # Add content before this chapter
        if last_end < match.start():
            chapters.append(('', full_text[last_end:match.start()]))
        
        ordinal = match.group(1).strip()
        chapter_num = vietnamese_ordinal_to_number(ordinal)
        
        # Find the end of this chapter (start of next chapter or end of text)
        next_match = None
        for next_match_candidate in re.finditer(chapter_pattern, full_text[match.end():]):
            next_match = next_match_candidate
            break
        
        if next_match:
            chapter_end = match.end() + next_match.start()
        else:
            chapter_end = len(full_text)
        
        chapter_content = full_text[match.start():chapter_end].strip()
        chapters.append((ordinal, chapter_content))
        last_end = chapter_end
    
    if chapters:
        for ordinal, chapter_content in chapters:
            if ordinal and chapter_content:  # Skip empty chapters
                chapter_num = vietnamese_ordinal_to_number(ordinal)
                
                # Extract chapter title from first line
                lines = chapter_content.split('\n')
                first_line = lines[0] if lines else ""
                title_match = re.search(r'Hồi thứ [^.]*[.:]?\s*(.*)', first_line)
                chapter_title = title_match.group(1).strip() if title_match else ""
                
                filename = f"Hồi_thứ_{chapter_num}_{ordinal}_{chapter_title}.txt"
                # Clean filename
                filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                filename = re.sub(r'_+', '_', filename)  # Remove multiple underscores
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(chapter_content)
                
                print(f"Extracted: {filename}")
    else:
        print("No chapter markers found. Saving as single file.")
        with open(os.path.join(output_dir, "full_text.txt"), 'w', encoding='utf-8') as f:
            f.write(full_text)
    
    return True

def extract_chapters_with_pypdf2():
    """Extract chapters using PyPDF2 library"""
    try:
        import PyPDF2
    except ImportError:
        print("PyPDF2 not available. Please install with: pip install PyPDF2")
        return False
    
    pdf_path = "viet_text.pdf"
    output_dir = "vietnamese"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Open the PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        print(f"PDF has {len(pdf_reader.pages)} pages")
        
        # Extract text from page 3 onwards (skip first 2 pages)
        full_text = ""
        for page_num in range(2, len(pdf_reader.pages)):  # Start from page 3 (index 2)
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            full_text += text + "\n"
    
    # Split into chapters based on chapter markers
    # Look for patterns like "Hồi thứ nhất", "Hồi thứ hai", etc.
    chapter_pattern = r'Hồi thứ ([^.\n]*?)(?:\.|:|\n)'
    chapter_matches = re.finditer(chapter_pattern, full_text)
    
    chapters = []
    last_end = 0
    
    for match in chapter_matches:
        # Add content before this chapter
        if last_end < match.start():
            chapters.append(('', full_text[last_end:match.start()]))
        
        ordinal = match.group(1).strip()
        chapter_num = vietnamese_ordinal_to_number(ordinal)
        
        # Find the end of this chapter (start of next chapter or end of text)
        next_match = None
        for next_match_candidate in re.finditer(chapter_pattern, full_text[match.end():]):
            next_match = next_match_candidate
            break
        
        if next_match:
            chapter_end = match.end() + next_match.start()
        else:
            chapter_end = len(full_text)
        
        chapter_content = full_text[match.start():chapter_end].strip()
        chapters.append((ordinal, chapter_content))
        last_end = chapter_end
    
    if chapters:
        for ordinal, chapter_content in chapters:
            if ordinal and chapter_content:  # Skip empty chapters
                chapter_num = vietnamese_ordinal_to_number(ordinal)
                
                # Extract chapter title from first line
                lines = chapter_content.split('\n')
                first_line = lines[0] if lines else ""
                title_match = re.search(r'Hồi thứ [^.]*[.:]?\s*(.*)', first_line)
                chapter_title = title_match.group(1).strip() if title_match else ""
                
                filename = f"Hồi_thứ_{chapter_num}_{ordinal}_{chapter_title}.txt"
                # Clean filename
                filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                filename = re.sub(r'_+', '_', filename)  # Remove multiple underscores
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(chapter_content)
                
                print(f"Extracted: {filename}")
    else:
        print("No chapter markers found. Saving as single file.")
        with open(os.path.join(output_dir, "full_text.txt"), 'w', encoding='utf-8') as f:
            f.write(full_text)
    
    return True

def main():
    """Main function to extract chapters"""
    print("Extracting chapters from HoangLeNhatThongChi Vietnamese PDF...")
    
    # Try PyMuPDF first (generally better text extraction)
    if extract_chapters_with_pymupdf():
        print("Successfully extracted chapters using PyMuPDF")
    elif extract_chapters_with_pypdf2():
        print("Successfully extracted chapters using PyPDF2")
    else:
        print("No PDF processing library available. Please install PyMuPDF or PyPDF2:")
        print("pip install PyMuPDF")
        print("or")
        print("pip install PyPDF2")

if __name__ == "__main__":
    main() 