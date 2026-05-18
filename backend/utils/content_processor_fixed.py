import os
from typing import Dict
import logging
import requests

logger = logging.getLogger(__name__)


class GemmaContentProcessor:
    """
    Processes learning content using local Ollama + RAG pipeline.
    """

    def __init__(
        self,
        model_path: str,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "eduvia:latest"
    ):
        self.model_path = model_path
        self.ollama_url = ollama_url
        self.model_name = model_name
        self._is_loaded = False
        self._rag_pipeline = None

    # =========================================================
    # RAG
    # =========================================================

    def _get_rag_pipeline(self):
        """Lazy-load RAG pipeline."""
        if self._rag_pipeline is None:
            from utils.rag_pipeline import get_rag_pipeline
            self._rag_pipeline = get_rag_pipeline()

        return self._rag_pipeline

    # =========================================================
    # MODEL HEALTH CHECK
    # =========================================================

    def _check_ollama_reachable(self) -> bool:
        """Quick check if Ollama is reachable (fails fast)."""
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=3
            )
            return response.status_code == 200
        except Exception:
            return False

    def load_model(self) -> bool:
        """
        Simple Ollama health check.
        """
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=5
            )

            if response.status_code == 200:
                logger.info("✅ Ollama server reachable")
                self._is_loaded = True
                return True

            logger.error("❌ Ollama server returned non-200 status")
            return False

        except Exception as e:
            logger.error(f"❌ Ollama health check failed: {e}")
            return False

    # =========================================================
    # GENERATION
    # =========================================================

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512
    ) -> str:
        """
        Generate text from Ollama model.
        """

        # Quick reachability check – fail fast if Ollama is down
        if not self._check_ollama_reachable():
            logger.error(f"❌ Ollama not reachable at {self.ollama_url}")
            return (
                "⚠️ Ollama is not running. Please start Ollama first:\n"
                "1. Open a terminal and run: ollama serve\n"
                "2. Then run: ollama run eduvia:latest\n"
                "3. Try your question again."
            )

        try:
            # (connect_timeout=10, read_timeout=300)
            # Connect should be instant since we verified above;
            # read can be slow for large model inference.
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens
                    }
                },
                timeout=(10, 1200)  # Increased timeout for slow CPU inference
            )

            response.raise_for_status()

            data = response.json()

            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Generation failed - Ollama not reachable at {self.ollama_url}: {e}")
            return "Ollama service is not running. Please start Ollama and ensure the model 'eduvia' is loaded."
        except requests.exceptions.Timeout as e:
            logger.error(f"❌ Generation timed out: {e}")
            return "Response generation timed out. Please try again."
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return "I'm having trouble generating a response right now."

    # =========================================================
    # CONTENT PROCESSING
    # =========================================================

    def process_content(self, text: str, task: str = "full") -> Dict:
        """
        Process content using Gemma 4 model.
        
        Tasks:
        - extract: Extract key concepts and sections
        - summarize: Generate content summary
        - simplify: Simplify explanations
        - structure: Create learning chunks
        - teach_styles: Generate multiple teaching styles
        - full: Run all processing methods
        """
        try:
            if task == "extract":
                prompt = f"""Extract key concepts, main topics, and important sections from the following text.

Text:
{text[:2000]}

Provide output as a structured list with concepts and sections."""
                result = self.generate(prompt)
                return {
                    "key_concepts": result.split("\n")[:10],
                    "sections": []
                }
            
            elif task == "summarize":
                prompt = f"""Provide a detailed, topic-by-topic summary of the following document. Break it down into clear logical chunks or sections.

Document Text:
{text[:15000]}

Please ensure the summary covers all main topics discussed, with bullet points or numbered lists where appropriate."""
                result = self.generate(prompt, max_new_tokens=2048)
                return {"summary": result}
            
            elif task == "simplify":
                prompt = f"""Simplify and explain the following text in easier language:

{text[:2000]}

Make it easy to understand for beginners."""
                result = self.generate(prompt)
                return {"simplified": result}
            
            elif task == "structure":
                prompt = f"""Break down the following text into logical learning chunks or sections:

{text[:2000]}

Organize it as step-by-step learning chunks."""
                result = self.generate(prompt)
                return {"learning_chunks": result.split("\n\n")}
            
            elif task == "teach_styles":
                prompt = f"""Present the following content in multiple teaching styles (Socratic, Narrative, Step-by-step):

{text[:2000]}

Provide 3 different teaching approaches."""
                result = self.generate(prompt)
                return {"teaching_styles": result.split("\n---\n")}
            
            elif task == "full":
                # Run all processing tasks
                results = {}
                for subtask in ["extract", "summarize", "simplify", "structure"]:
                    results[subtask] = self.process_content(text, subtask)
                return results
            
            else:
                return {"error": f"Unknown task: {task}"}
        
        except Exception as e:
            logger.error(f"Content processing failed for task '{task}': {e}")
            return {"error": str(e), "task": task}

    # =========================================================
    # TUTOR INTERACTION
    # =========================================================

    def tutor_interaction(
        self,
        query: str,
        learning_state: str = "focused",
        doc_name: str = "default"
    ) -> Dict:
        """
        Full tutor interaction with:
        - RAG retrieval
        - learning-state adaptation
        - Ollama generation
        """

        rag = self._get_rag_pipeline()

        # -----------------------------------------------------
        # Check indexed docs
        # -----------------------------------------------------

        if (
            not hasattr(rag, "indexed_docs")
            or doc_name not in rag.indexed_docs
        ):
            # No document indexed - but we can still answer using general knowledge
            logger.info(f"ℹ️ No document indexed, using general knowledge for: {query}")
            
            # Use general knowledge prompt
            prompt = f"""You are Eduvia, an adaptive AI tutor helping a student.

STUDENT'S LEARNING STATE: {learning_state.upper()}

STUDENT'S QUESTION: {query}

Please provide a helpful and clear answer using your knowledge."""

            answer = self.generate(prompt, max_new_tokens=1024)
            
            if not answer or len(answer.strip()) < 10:
                answer = "I'm having trouble generating a response. Please try again."
            
            return {
                "query": query,
                "answer": answer,
                "learning_state": learning_state,
                "sources": [],
                "rag_enabled": False,
                "context_used": False,
                "document_indexed": False,
                "message": "Answered using general knowledge (no document indexed)"
            }

        # -----------------------------------------------------
        # Retrieve context
        # -----------------------------------------------------

        rag_result = rag.query(query, doc_name)

        context = rag_result.get("context", "")
        sources = rag_result.get("sources", [])

        has_meaningful_context = (
            bool(context)
            and len(context.strip()) > 50
        )

        # -----------------------------------------------------
        # Adaptive instructions
        # -----------------------------------------------------

        state_instructions = {
            "focused":
                "Provide detailed and comprehensive explanations.",

            "distracted":
                "Keep explanations short, engaging, and easy to follow.",

            "overloaded":
                (
                    "Break concepts into small steps. "
                    "Explain one idea at a time."
                ),

            "low_engagement":
                (
                    "Use an engaging conversational style "
                    "with questions and examples."
                )
        }

        instruction = state_instructions.get(
            learning_state,
            state_instructions["focused"]
        )

        # -----------------------------------------------------
        # Prompt construction
        # -----------------------------------------------------

        if has_meaningful_context:

            prompt = f"""You are Eduvia, an adaptive AI tutor helping a student.

STUDENT'S LEARNING STATE: {learning_state.upper()}
TEACHING APPROACH: {instruction}

REFERENCE MATERIAL FROM THE DOCUMENT:
{context}

STUDENT'S QUESTION: {query}

Please answer the student's question based on the reference material provided above. 
If the answer is found in the reference material, use that information.
Be clear, accurate, and educational."""

        else:

            prompt = f"""You are Eduvia, an adaptive AI tutor helping a student.

STUDENT'S LEARNING STATE: {learning_state.upper()}
TEACHING APPROACH: {instruction}

STUDENT'S QUESTION: {query}

Please provide a helpful answer to the student's question.
Explain the concept clearly in an educational way."""

        # -----------------------------------------------------
        # Generate response
        # -----------------------------------------------------

        logger.info(f"📝 Generated tutor prompt | Context: {has_meaningful_context} | Learner state: {learning_state}")

        answer = self.generate(prompt, max_new_tokens=1024)

        if not answer or len(answer.strip()) < 10:
            answer = (
                "I'm having trouble generating a response right now. "
                "Please try again or rephrase your question."
            )

        return {
            "query": query,
            "answer": answer,
            "learning_state": learning_state,
            "sources": sources,
            "rag_enabled": True,
            "context_used": has_meaningful_context,
            "document_indexed": True
        }

    # =========================================================
    # DOCUMENT INDEXING
    # =========================================================

    def index_document_for_rag(
        self,
        text: str,
        doc_name: str = "default"
    ) -> Dict:
        """
        Index document into RAG system.
        """

        rag = self._get_rag_pipeline()

        chunks = rag.load_document(text, doc_name)

        return {
            "status": "indexed",
            "document": doc_name,
            "num_chunks": len(chunks),
            "message": (
                f"✅ Indexed {len(chunks)} chunks for retrieval"
            )
        }


# =============================================================
# Singleton
# =============================================================

_processor_instance = None


def get_content_processor(
    model_path: str = None
) -> GemmaContentProcessor:

    global _processor_instance

    if _processor_instance is None:

        if model_path is None:
            backend_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

            model_path = os.path.join(
                backend_dir,
                "gemma4_merged"
            )

        _processor_instance = GemmaContentProcessor(model_path)

        logger.info(
            f"📦 Created content processor "
            f"with model path: {model_path}"
        )

    return _processor_instance