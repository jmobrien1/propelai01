"""
Text Cleaning Utilities
Handles HTML entities, special characters, and text normalization
"""

import re
import html
from typing import str


class TextCleaner:
    """Clean and normalize text extracted from documents"""
    
    # HTML entities that commonly appear in extracted text
    HTML_ENTITY_MAP = {
        '&#8220;': '"',  # Left double quote
        '&#8221;': '"',  # Right double quote
        '&#8217;': "'",  # Right single quote
        '&#8216;': "'",  # Left single quote
        '&#8212;': '—',  # Em dash
        '&#8211;': '–',  # En dash
        '&#8230;': '...',  # Ellipsis
        '&nbsp;': ' ',   # Non-breaking space
        '&amp;': '&',    # Ampersand
        '&lt;': '<',     # Less than
        '&gt;': '>',     # Greater than
        '&quot;': '"',   # Quote
        '&apos;': "'",   # Apostrophe
    }
    
    # Unicode replacements for common issues
    UNICODE_REPLACEMENTS = {
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2013': '-',   # En dash
        '\u2014': '--',  # Em dash
        '\u2026': '...', # Ellipsis
        '\xa0': ' ',     # Non-breaking space
        '\u00a0': ' ',   # Another non-breaking space
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Comprehensive text cleaning
        
        Steps:
        1. Decode HTML entities (both named and numeric)
        2. Replace Unicode special characters
        3. Normalize whitespace
        4. Remove control characters
        5. Fix common OCR errors
        
        Args:
            text: Raw text from document parser
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Step 1: HTML entity decoding
        text = TextCleaner._decode_html_entities(text)
        
        # Step 2: Unicode character replacement
        text = TextCleaner._replace_unicode_chars(text)
        
        # Step 3: Normalize whitespace
        text = TextCleaner._normalize_whitespace(text)
        
        # Step 4: Remove control characters (except newlines and tabs)
        text = TextCleaner._remove_control_chars(text)
        
        # Step 5: Fix common OCR errors
        text = TextCleaner._fix_ocr_errors(text)
        
        return text.strip()
    
    @staticmethod
    def _decode_html_entities(text: str) -> str:
        """Decode HTML entities"""
        # First decode named entities using html.unescape
        text = html.unescape(text)
        
        # Then handle any remaining numeric entities manually
        for entity, replacement in TextCleaner.HTML_ENTITY_MAP.items():
            text = text.replace(entity, replacement)
        
        # Decode any remaining &#NNNN; patterns
        def decode_numeric_entity(match):
            try:
                num = int(match.group(1))
                return chr(num)
            except (ValueError, OverflowError):
                return match.group(0)
        
        text = re.sub(r'&#(\d+);', decode_numeric_entity, text)
        
        return text
    
    @staticmethod
    def _replace_unicode_chars(text: str) -> str:
        """Replace problematic Unicode characters"""
        for unicode_char, replacement in TextCleaner.UNICODE_REPLACEMENTS.items():
            text = text.replace(unicode_char, replacement)
        return text
    
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace while preserving paragraph structure"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at line ends
        text = re.sub(r' +\n', '\n', text)
        
        # Remove spaces at line starts (but preserve intentional indentation)
        text = re.sub(r'\n +', '\n', text)
        
        return text
    
    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """Remove control characters except newlines, tabs, and carriage returns"""
        # Remove control characters except \n, \t, \r
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text
    
    @staticmethod
    def _fix_ocr_errors(text: str) -> str:
        """Fix common OCR errors"""
        # Fix common OCR substitutions
        ocr_fixes = {
            r'\bO(?=\d)': '0',  # Letter O before digit -> zero
            r'(?<=\d)O\b': '0',  # Letter O after digit -> zero
            r'\bl(?=\d)': '1',   # Lowercase L before digit -> one
            r'(?<=\d)l\b': '1',  # Lowercase L after digit -> one
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @staticmethod
    def clean_requirement_text(text: str) -> str:
        """
        Clean requirement text specifically
        Additional cleaning beyond general text cleaning
        """
        text = TextCleaner.clean_text(text)
        
        # Remove excessive punctuation
        text = re.sub(r'\.{4,}', '...', text)  # Multiple dots -> ellipsis
        text = re.sub(r'-{3,}', '--', text)    # Multiple dashes -> double dash
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])(?=[A-Za-z])', r'\1 ', text)  # Add space after punctuation
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        return text
    
    @staticmethod
    def extract_plain_text(text: str) -> str:
        """
        Extract plain text, removing formatting markers
        Useful for analysis while preserving cleaned content
        """
        text = TextCleaner.clean_text(text)
        
        # Remove markdown-style formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
        text = re.sub(r'__(.+?)__', r'\1', text)      # Bold
        text = re.sub(r'_(.+?)_', r'\1', text)        # Italic
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        return text.strip()


def clean_text(text: str) -> str:
    """Convenience function for text cleaning"""
    return TextCleaner.clean_text(text)


def clean_requirement_text(text: str) -> str:
    """Convenience function for requirement text cleaning"""
    return TextCleaner.clean_requirement_text(text)
