"""
RAG Document Chunking & Retrieval System
Chunks PDF content and enables semantic retrieval for tutor responses.
"""

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of document with metadata for retrieval."""
    chunk_id: str
    content: str
    source: str
    page_number: int
    chunk_index: int
    heading: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


class DocumentChunker:
    """Splits documents into semantic chunks for RAG."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target tokens per chunk
            overlap: Overlap between chunks for context continuity
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source: str = "document") -> List[DocumentChunk]:
        """
        Split text into semantic chunks.
        
        Uses paragraph boundaries and sentence boundaries to create
        meaningful chunks that preserve context.
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        page_number = 1
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Estimate tokens (rough: 4 chars per token)
            para_tokens = len(para) // 4
            
            # If single paragraph is too large, split it
            if para_tokens > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, source, page_number, chunk_index
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_chunk = ""
                
                # Split large paragraph by sentences
                sentences = self._split_sentences(para)
                temp_chunk = ""
                
                for sent in sentences:
                    sent_tokens = len(sent) // 4
                    if len(temp_chunk) // 4 + sent_tokens > self.chunk_size:
                        chunk = self._create_chunk(
                            temp_chunk, source, page_number, chunk_index
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1
                        
                        # Keep overlap
                        if self.overlap > 0 and temp_chunk:
                            overlap_text = temp_chunk[-self.overlap * 4:]
                            temp_chunk = overlap_text + " " + sent
                        else:
                            temp_chunk = sent
                    else:
                        temp_chunk += " " + sent if temp_chunk else sent
                
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                # Check if adding this paragraph exceeds chunk size
                current_tokens = len(current_chunk) // 4
                
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    # Create chunk from accumulated content
                    chunk = self._create_chunk(
                        current_chunk, source, page_number, chunk_index
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    # Keep overlap
                    if self.overlap > 0:
                        overlap_chars = min(self.overlap * 4, len(current_chunk))
                        current_chunk = current_chunk[-overlap_chars:]
                    else:
                        current_chunk = ""
                
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Don't forget last chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk, source, page_number, chunk_index
            )
            if chunk:
                chunks.append(chunk)
        
        logger.info(f"✅ Created {len(chunks)} chunks from document")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(self, content: str, source: str, page: int, index: int) -> Optional[DocumentChunk]:
        """Create a DocumentChunk with generated metadata."""
        if len(content.strip()) < 50:  # Skip too small chunks
            return None
        
        # Generate unique ID
        chunk_id = hashlib.md5(
            f"{source}:{page}:{index}:{content[:50]}".encode()
        ).hexdigest()[:12]
        
        # Extract keywords (simple approach)
        keywords = self._extract_keywords(content)
        
        # Generate summary (first sentence(s))
        summary = self._generate_summary(content)
        
        # Try to detect heading
        heading = self._detect_heading(content)
        
        return DocumentChunk(
            chunk_id=chunk_id,
            content=content.strip(),
            source=source,
            page_number=page,
            chunk_index=index,
            heading=heading,
            summary=summary,
            keywords=keywords
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        import re
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Common stop words to exclude
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 
                      'will', 'their', 'what', 'about', 'which', 'when',
                      'make', 'like', 'time', 'just', 'know', 'take',
                      'people', 'into', 'year', 'your', 'good', 'some',
                      'could', 'them', 'see', 'other', 'than', 'then',
                      'now', 'look', 'only', 'come', 'its', 'over'}
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:10]]
    
    def _generate_summary(self, text: str) -> str:
        """Generate a brief summary of the chunk."""
        # Use first 1-2 sentences as summary
        sentences = self._split_sentences(text)
        if len(sentences) > 1:
            return sentences[0][:200] + "..." if len(sentences[0]) > 200 else sentences[0]
        return text[:200] + "..." if len(text) > 200 else text
    
    def _detect_heading(self, text: str) -> Optional[str]:
        """Detect if chunk starts with a heading."""
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if it looks like a heading (short, no ending punctuation, capitalized)
            if (len(first_line) < 100 and 
                not first_line.endswith('.') and
                first_line[0].isupper() if first_line else False):
                return first_line
        return None


class SimpleRetriever:
    """Simple keyword-based retriever for document chunks."""
    
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: Dict[str, List[float]] = {}
    
    def index_chunks(self, chunks: List[DocumentChunk]):
        """Index chunks for retrieval."""
        self.chunks = chunks
        logger.info(f"📚 Indexed {len(chunks)} chunks for retrieval")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Uses keyword matching + position boosting for simple retrieval.
        For production, would use semantic embeddings (sentence-transformers).
        """
        query_words = query.lower().split()
        query_set = set(query_words)
        
        # Score each chunk
        chunk_scores = []
        
        for chunk in self.chunks:
            score = 0.0
            
            # Keyword matching in content
            chunk_content = chunk.content.lower()
            for word in query_set:
                if word in chunk_content:
                    score += 1.0
                    # Bonus for keyword match
                    if word in chunk.keywords:
                        score += 2.0
            
            # Boost for heading match
            if chunk.heading and any(word in chunk.heading.lower() for word in query_set):
                score += 3.0
            
            # Boost for summary match
            if chunk.summary:
                summary_lower = chunk.summary.lower()
                for word in query_set:
                    if word in summary_lower:
                        score += 1.5
            
            # Boost for earlier chunks (often more important)
            if chunk.chunk_index < 3:
                score *= 1.2
            
            if score > 0:
                chunk_scores.append((chunk, score))
        
        # Sort by score and return top_k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return chunk_scores[:top_k]
    
    def get_context_for_query(self, query: str, max_chars: int = 3000) -> str:
        """
        Get concatenated context from top chunks for RAG prompt.
        
        This is what gets injected into the LLM prompt as context.
        """
        results = self.retrieve(query, top_k=5)
        
        context_parts = []
        total_chars = 0
        
        for chunk, score in results:
            if total_chars + len(chunk.content) > max_chars:
                break
            context_parts.append(f"[Page {chunk.page_number}, Chunk {chunk.chunk_index + 1}]\n{chunk.content}")
            total_chars += len(chunk.content)
        
        return "\n\n".join(context_parts)


class RAGPipeline:
    """Complete RAG pipeline for tutor interactions."""
    
    def __init__(self):
        self.chunker = DocumentChunker(chunk_size=500, overlap=50)
        self.retriever = SimpleRetriever()
        self.indexed_docs: Dict[str, List[DocumentChunk]] = {}
    
    def load_document(self, text: str, doc_name: str = "default") -> List[DocumentChunk]:
        """Load and chunk a document."""
        chunks = self.chunker.chunk_text(text, source=doc_name)
        self.indexed_docs[doc_name] = chunks
        self.retriever.index_chunks(chunks)
        return chunks
    
    def query(self, query: str, doc_name: str = "default", top_k: int = 5) -> Dict:
        """
        Query the document with RAG.
        
        Returns:
            Dict with 'context', 'sources', 'answer' (to be generated by LLM)
        """
        if doc_name not in self.indexed_docs:
            return {
                "error": f"Document '{doc_name}' not loaded",
                "context": "",
                "sources": []
            }
        
        # Retrieve relevant chunks
        results = self.retriever.retrieve(query, top_k=top_k)
        
        # Build context
        context = self.retriever.get_context_for_query(query, max_chars=3000)
        
        # Build source references
        sources = []
        for chunk, score in results:
            sources.append({
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_number,
                "summary": chunk.summary,
                "relevance_score": round(score, 2)
            })
        
        return {
            "query": query,
            "context": context,
            "sources": sources,
            "num_chunks_retrieved": len(results)
        }
    
    def get_rag_prompt(self, query: str, doc_name: str = "default") -> str:
        """
        Generate RAG-enhanced prompt for the LLM.
        
        This creates the prompt with retrieved context injected.
        """
        rag_result = self.query(query, doc_name)
        
        prompt = f"""<start_of_turn>user
You are an expert tutor. Use the provided document context to answer the user's question accurately.

DOCUMENT CONTEXT:
{rag_result['context']}

Based on the above context from the document, please answer this question:
{query}

If the context doesn't contain enough information to fully answer, please say so and use what information is available.
<end_of_turn>
<start_of_turn>model
"""
        return prompt


# Singleton instance
_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create singleton RAG pipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline