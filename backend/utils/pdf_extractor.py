"""
PDF Text Extraction Module
Extracts clean text from PDF files and prepares for AI processing.
"""

import PyPDF2
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts and cleans text from PDF files."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.clean_text = ""
        self.metadata = {}
        self.pages = []
    
    def extract(self) -> Dict:
        """
        Extract text from PDF and return structured data.
        
        Returns:
            Dict with 'raw_text', 'clean_text', 'metadata', 'pages'
        """
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                self.metadata = {
                    'title': reader.metadata.get('/Title', 'Unknown'),
                    'author': reader.metadata.get('/Author', 'Unknown'),
                    'subject': reader.metadata.get('/Subject', 'Unknown'),
                    'creator': reader.metadata.get('/Creator', 'Unknown'),
                    'producer': reader.metadata.get('/Producer', 'Unknown'),
                    'page_count': len(reader.pages)
                }
                
                # Extract text from each page
                self.pages = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    self.pages.append({
                        'page_number': i + 1,
                        'text': page_text
                    })
                    self.raw_text += page_text + "\n\n"
                
                # Clean the extracted text
                self.clean_text = self._clean_text(self.raw_text)
                
                logger.info(f"✅ Extracted {len(self.pages)} pages from {self.pdf_path}")
                
                return {
                    'raw_text': self.raw_text,
                    'clean_text': self.clean_text,
                    'metadata': self.metadata,
                    'pages': self.pages
                }
                
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing noise and artifacts.
        
        - Removes excessive whitespace
        - Fixes broken words
        - Removes special characters
        - Normalizes line breaks
        """
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove headers/footers (lines with only special chars)
        text = re.sub(r'^[-=_\u2500]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        # Fix common PDF extraction artifacts
        text = re.sub(r'(?<=[a-z])-\s+(?=[a-z])', '', text)  # Fix hyphenated words
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '--')
        
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove empty lines at start/end
        text = text.strip()
        
        return text
    
    def get_page(self, page_num: int) -> Optional[str]:
        """Get text from a specific page."""
        if 0 < page_num <= len(self.pages):
            return self.pages[page_num - 1]['text']
        return None
    
    def get_text_by_range(self, start_page: int, end_page: int) -> str:
        """Get combined text from a range of pages."""
        texts = []
        for i in range(start_page - 1, min(end_page, len(self.pages))):
            texts.append(self.pages[i]['text'])
        return '\n\n'.join(texts)