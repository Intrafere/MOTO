"""
Extract metadata from document chunks.
"""
from typing import Dict, Any
import re


def extract_chunk_metadata(
    text: str,
    source_file: str,
    position: int,
    chunk_size: int
) -> Dict[str, Any]:
    """
    Extract metadata from a chunk.
    
    Returns:
        Dict with metadata including:
        - source_file: original file
        - position: position in document
        - chunk_size: size configuration used
        - char_count: actual character count
        - word_count: word count
        - sentence_count: sentence count
        - chunk_type: detected type (text, code, table, equation)
    """
    metadata = {
        'source_file': source_file,
        'position': position,
        'chunk_size': chunk_size,
        'char_count': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'chunk_type': 'text'
    }
    
    # Detect chunk type
    if is_code_chunk(text):
        metadata['chunk_type'] = 'code'
    elif is_table_chunk(text):
        metadata['chunk_type'] = 'table'
    elif is_equation_chunk(text):
        metadata['chunk_type'] = 'equation'
    elif is_section_header(text):
        metadata['chunk_type'] = 'section'
    
    return metadata


def is_code_chunk(text: str) -> bool:
    """Detect if chunk is primarily code."""
    code_indicators = [
        r'def\s+\w+\s*\(',
        r'class\s+\w+',
        r'function\s+\w+\s*\(',
        r'import\s+\w+',
        r'=>',
        r'{\s*$',
        r'}\s*$',
    ]
    matches = sum(1 for pattern in code_indicators if re.search(pattern, text))
    return matches >= 2


def is_table_chunk(text: str) -> bool:
    """Detect if chunk is a table."""
    lines = text.split('\n')
    table_lines = sum(1 for line in lines if '|' in line or '\t' in line)
    return table_lines >= 3


def is_equation_chunk(text: str) -> bool:
    """Detect if chunk contains equations."""
    equation_patterns = [
        r'\$.*\$',
        r'\\\[.*\\\]',
        r'\\begin\{equation\}',
        r'\\frac\{',
        r'\\sum',
        r'\\int',
    ]
    return any(re.search(pattern, text) for pattern in equation_patterns)


def is_section_header(text: str) -> bool:
    """Detect if chunk is a section header."""
    # Check for markdown headers or short capitalized text
    if re.match(r'^#{1,6}\s+', text):
        return True
    if len(text) < 100 and text.isupper():
        return True
    if len(text) < 100 and re.match(r'^\d+\.?\s+[A-Z]', text):
        return True
    return False

