"""
Text normalization for document ingestion.
"""
import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    - Normalize unicode
    - Fix common encoding issues
    - Standardize whitespace
    - Preserve structure (paragraphs, sentences)
    """
    if not text:
        return ""
    
    # Normalize unicode to NFC form
    text = unicodedata.normalize('NFC', text)
    
    # Fix common encoding issues
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    
    # Standardize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Standardize dashes
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    # Normalize whitespace within lines (but preserve paragraph breaks)
    lines = text.split('\n')
    normalized_lines = []
    for line in lines:
        # Collapse multiple spaces to single space
        line = re.sub(r'[ \t]+', ' ', line)
        line = line.strip()
        normalized_lines.append(line)
    
    # Join lines, preserving paragraph breaks (empty lines)
    text = '\n'.join(normalized_lines)
    
    # Normalize multiple newlines to at most 2 (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_metadata_from_text(text: str) -> dict:
    """
    Extract metadata from text content.
    Returns dict with detected properties.
    """
    metadata = {}
    
    # Detect if text contains code
    code_patterns = [
        r'def\s+\w+\s*\(',
        r'class\s+\w+',
        r'function\s+\w+\s*\(',
        r'import\s+\w+',
        r'#include\s*<',
    ]
    metadata['has_code'] = any(re.search(pattern, text) for pattern in code_patterns)
    
    # Detect if text contains tables
    table_patterns = [
        r'\|.*\|.*\|',  # Markdown table
        r'\t.*\t.*\t',  # Tab-separated
    ]
    metadata['has_table'] = any(re.search(pattern, text) for pattern in table_patterns)
    
    # Detect if text contains equations
    equation_patterns = [
        r'\$.*\$',  # LaTeX inline
        r'\\\[.*\\\]',  # LaTeX block
        r'\\begin\{equation\}',
    ]
    metadata['has_equation'] = any(re.search(pattern, text) for pattern in equation_patterns)
    
    # Detect language (simple heuristic)
    # Count non-ASCII characters
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    metadata['is_ascii'] = non_ascii_count < len(text) * 0.05
    
    return metadata

