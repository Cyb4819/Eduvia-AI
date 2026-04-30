from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import time

router = APIRouter()

# Default PDF path
DEFAULT_PDF_PATH = r"C:\Users\somos\OneDrive\Desktop\Future of Education - Eduvia AI\jesc101.pdf"

# Content processor singleton
_content_processor = None


def get_content_processor():
    """Get or initialize content processor."""
    global _content_processor
    if _content_processor is None:
        from utils.content_processor import get_content_processor
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
        _content_processor = get_content_processor(model_path)
    return _content_processor


class PDFLoadRequest(BaseModel):
    pdf_path: Optional[str] = None


class ContentProcessRequest(BaseModel):
    text: str
    task: str = "full"  # extract, summarize, simplify, structure, teach_styles, full


@router.get("/health")
def health_check():
    """Minimal health check for Phase 1."""
    return {
        "status": "✅ Phase 1 Behavioral Observer ready",
        "observer_path": "backend/observation/run_observer.bat",
        "signals": ["presence", "gaze_on_screen", "movement_level", "attention_state"],
        "learning_states": ["focused", "distracted", "overloaded", "low_engagement"]
    }

@router.get("/learning-states/info")
def learning_states_info():
    """Get information about available learning states."""
    return {
        "learning_states": {
            "focused": {
                "description": "Learner is present, looking at screen, with low movement",
                "indicators": ["presence=true", "gaze_on_screen=true", "movement_level=low"],
                "teaching_strategy": "Continue detailed explanation"
            },
            "distracted": {
                "description": "Learner looking away from screen for extended periods",
                "indicators": ["presence=true", "gaze_on_screen=false", "duration>=5s"],
                "teaching_strategy": "Simplify and shorten content"
            },
            "overloaded": {
                "description": "High movement with low focus, signs of cognitive overload",
                "indicators": ["presence=true", "movement_level=high", "gaze_on_screen=false"],
                "teaching_strategy": "Break into step-by-step guidance"
            },
            "low_engagement": {
                "description": "Absent or inactive for extended periods",
                "indicators": ["presence=false", "or presence=true but inactivity>=10s"],
                "teaching_strategy": "Rephrase or change explanation style"
            }
        },
        "rule_based_thresholds": {
            "gaze_away_distracted": "5 seconds",
            "movement_high": ">15 pixels",
            "inactivity_low_engagement": "10 seconds"
        }
    }


# ============================================
# Phase 3: Learning Content Processing APIs
# ============================================

@router.post("/content/load-pdf")
def load_pdf(request: PDFLoadRequest):
    """
    Load and extract text from a PDF file.
    Preloads the PDF and prepares it for processing.
    """
    pdf_path = request.pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")
    
    try:
        from utils.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(pdf_path)
        result = extractor.extract()
        
        return {
            "status": "success",
            "pdf_path": pdf_path,
            "metadata": result['metadata'],
            "page_count": len(result['pages']),
            "text_length": len(result['clean_text']),
            "message": f"✅ Loaded {len(result['pages'])} pages from PDF"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF loading failed: {str(e)}")


@router.get("/content/pdf/{page_num}")
def get_pdf_page(page_num: int, pdf_path: Optional[str] = None):
    """Get text from a specific page of the loaded PDF."""
    target_path = pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"PDF not found")
    
    try:
        from utils.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(target_path)
        extractor.extract()
        
        page_text = extractor.get_page(page_num)
        if page_text is None:
            raise HTTPException(status_code=404, detail=f"Page {page_num} not found")
        
        return {
            "page": page_num,
            "text": page_text,
            "total_pages": len(extractor.pages)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content/pdf")
def get_full_pdf(pdf_path: Optional[str] = None):
    """Get the full extracted text from the PDF."""
    target_path = pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"PDF not found")
    
    try:
        from utils.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(target_path)
        result = extractor.extract()
        
        return {
            "status": "success",
            "metadata": result['metadata'],
            "full_text": result['clean_text'],
            "page_count": len(result['pages'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content/process")
def process_content(request: ContentProcessRequest):
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
        processor = get_content_processor()
        
        # Process content with specified task
        result = processor.process_content(request.text, request.task)
        
        return {
            "status": "success",
            "task": request.task,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/content/process-pdf")
def process_pdf(
    task: str = Query("full", description="Processing task: extract, summarize, simplify, structure, teach_styles, full"),
    pdf_path: Optional[str] = None
):
    """
    Load PDF and process with Gemma 4 in one call.
    """
    target_path = pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {target_path}")
    
    try:
        # Extract PDF text
        from utils.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(target_path)
        pdf_result = extractor.extract()
        
        # Process with Gemma 4
        processor = get_content_processor()
        processed = processor.process_content(pdf_result['clean_text'], task)
        
        return {
            "status": "success",
            "pdf_metadata": pdf_result['metadata'],
            "processing_task": task,
            "processed_content": processed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@router.get("/content/concepts")
def extract_concepts(pdf_path: Optional[str] = None):
    """Extract key concepts from PDF."""
    target_path = pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    
    try:
        from utils.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(target_path)
        pdf_result = extractor.extract()
        
        processor = get_content_processor()
        result = processor.process_content(pdf_result['clean_text'], "extract")
        
        return {
            "status": "success",
            "key_concepts": result.get('key_concepts', []),
            "sections": result.get('sections', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content/summary")
def get_summary(pdf_path: Optional[str] = None):
    """Get AI-generated summary of PDF content."""
    target_path = pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    
    try:
        from utils.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(target_path)
        pdf_result = extractor.extract()
        
        processor = get_content_processor()
        result = processor.process_content(pdf_result['clean_text'], "summarize")
        
        return {
            "status": "success",
            "summary": result.get('summary', 'No summary available')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content/chunks")
def get_learning_chunks(pdf_path: Optional[str] = None):
    """Get structured learning chunks from PDF."""
    target_path = pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    
    try:
        from utils.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(target_path)
        pdf_result = extractor.extract()
        
        processor = get_content_processor()
        result = processor.process_content(pdf_result['clean_text'], "structure")
        
        return {
            "status": "success",
            "learning_chunks": result.get('learning_chunks', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content/teaching-styles")
def get_teaching_styles(pdf_path: Optional[str] = None):
    """Get multiple teaching style versions of PDF content."""
    target_path = pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    
    try:
        from utils.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(target_path)
        pdf_result = extractor.extract()
        
        processor = get_content_processor()
        result = processor.process_content(pdf_result['clean_text'], "teach_styles")
        
        return {
            "status": "success",
            "teaching_styles": result.get('teaching_styles', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Phase 4: RAG-Based Tutor Interaction APIs
# ============================================

class TutorQuestion(BaseModel):
    question: str
    learning_state: str = "focused"


class IndexDocumentRequest(BaseModel):
    pdf_path: Optional[str] = None
    doc_name: str = "default"


# Store indexed document text for RAG
_indexed_documents: Dict[str, str] = {}


@router.post("/tutor/index-document")
def index_document_for_rag(request: IndexDocumentRequest):
    """
    Index a PDF document for RAG-based tutor interactions.
    
    This must be called BEFORE asking questions to enable
    the model to look at the document context.
    """
    target_path = request.pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {target_path}")
    
    try:
        # Extract PDF text
        from utils.pdf_extractor import PDFExtractor
        extractor = PDFExtractor(target_path)
        pdf_result = extractor.extract()
        
        # Get content processor and index for RAG
        processor = get_content_processor()
        index_result = processor.index_document_for_rag(
            pdf_result['clean_text'], 
            request.doc_name
        )
        
        # Store the text for later use
        _indexed_documents[request.doc_name] = pdf_result['clean_text']
        
        return {
            "status": "success",
            "message": f"✅ Document indexed for RAG. Ask me anything about it!",
            "document": request.doc_name,
            "chunks_indexed": index_result['num_chunks'],
            "text_length": len(pdf_result['clean_text'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/tutor/ask")
def tutorAsk(request: TutorQuestion):
    """
    Ask the tutor a question with RAG context.
    
    The model will look at the indexed document before answering.
    If the question is not in the document, it will still respond but acknowledge
    that the information is not from the loaded document.
    """
    try:
        processor = get_content_processor()
        
        # Get answer with RAG - the method handles both in-context and out-of-context questions
        result = processor.tutor_interaction(
            request.question,
            request.learning_state,
            "default"
        )
        
        return result
    except Exception as e:
        logger.error(f"Tutor error: {e}")
        raise HTTPException(status_code=500, detail=f"Tutor error: {str(e)}")


@router.get("/tutor/ask")
def tutorAskGet(
    question: str = Query(..., description="Your question about the material"),
    learning_state: str = Query("focused", description="Current learning state")
):
    """GET version of tutor ask endpoint."""
    return tutorAsk(TutorQuestion(question=question, learning_state=learning_state))


@router.get("/tutor/status")
def tutorStatus():
    """Get RAG indexing status."""
    return {
        "indexed_documents": list(_indexed_documents.keys()),
        "rag_active": len(_indexed_documents) > 0,
        "message": "RAG is active" if _indexed_documents else "No documents indexed"
    }


@router.get("/tutor/should-intervene")
def should_intervene(
    learning_state: str = Query("focused", description="Current learning state")
):
    """
    Check if the system should intervene based on learning state.
    
    Returns True for distracted, overloaded, low_engagement.
    Returns False for focused.
    """
    from utils.voice_trigger_controller import get_trigger_controller
    controller = get_trigger_controller()
    
    should = controller.should_respond_to_state(learning_state)
    
    return {
        "should_intervene": should,
        "learning_state": learning_state,
        "reason": "Non-focused state detected" if should else "User is focused - silent observation"
    }


@router.post("/tutor/voice-ask")
def voice_ask_post(request: TutorQuestion):
    """
    Voice-triggered Q&A (POST version).
    
    ALWAYS processes regardless of learning state.
    This is for explicit user voice input.
    """
    try:
        processor = get_content_processor()
        
        # Check if document is indexed
        if not _indexed_documents:
            return {
                "answer": "Please index a document first using /api/tutor/index-document",
                "sources": [],
                "rag_enabled": False
            }
        
        doc_name = "default"
        if doc_name not in _indexed_documents:
            return {
                "answer": "No document indexed. Call /api/tutor/index-document first.",
                "sources": [],
                "rag_enabled": False
            }
        
        # Get answer with RAG
        result = processor.tutor_interaction(
            request.question,
            request.learning_state,
            doc_name
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice tutor error: {str(e)}")


# ============================================
# Phase 5: Adaptive Learning Engine APIs
# ============================================

class ObserverSignals(BaseModel):
    """Observer signals from Phase 1."""
    presence: bool = True
    gaze_on_screen: bool = True
    movement_level: str = "low"
    attention_state: str = "focused"
    face_detected: bool = True
    timestamp: Optional[float] = None


class AdaptationRequest(BaseModel):
    """Request to force adaptation."""
    state: Optional[str] = None  # focused, distracted, overloaded, low_engagement


# Adaptive engine singleton
_adaptive_engine = None


def get_adaptive_engine():
    """Get or initialize adaptive engine."""
    global _adaptive_engine
    if _adaptive_engine is None:
        from utils.adaptive_engine import AdaptiveLearningEngine, AdaptationConfig
        config = AdaptationConfig(
            check_interval_seconds=5,
            adaptation_cooldown_seconds=30,
            auto_teach_enabled=True,
            auto_speak_enabled=True
        )
        _adaptive_engine = AdaptiveLearningEngine(config)
    return _adaptive_engine


@router.post("/adaptive/update-signals")
def update_observer_signals(signals: ObserverSignals):
    """
    Update adaptive engine with observer signals.
    
    This is the main entry point - Phase 1 observer calls this
    to feed signals into the adaptive learning loop.
    """
    try:
        engine = get_adaptive_engine()
        
        # Convert to dict
        signals_dict = {
            "presence": signals.presence,
            "gaze_on_screen": signals.gaze_on_screen,
            "movement_level": signals.movement_level,
            "attention_state": signals.attention_state,
            "face_detected": signals.face_detected,
            "timestamp": signals.timestamp or time.time()
        }
        
        # Update engine
        engine.update_observer_signals(signals_dict)
        
        # Get current state
        state = engine.get_current_state()
        
        return {
            "status": "success",
            "current_state": state.value,
            "signals_received": True,
            "adaptation_triggered": engine._should_adapt()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal update failed: {str(e)}")


@router.get("/adaptive/state")
def get_adaptive_state():
    """Get current adaptive engine state."""
    try:
        engine = get_adaptive_engine()
        summary = engine.get_state_summary()
        
        return {
            "status": "success",
            "current_learning_state": summary["current_state"],
            "previous_state": summary["previous_state"],
            "adaptation_count": summary["adaptation_count"],
            "strategy": summary["strategy"],
            "state_history_count": summary["state_history_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptive/start")
def start_adaptive_engine():
    """Start the adaptive learning loop."""
    try:
        engine = get_adaptive_engine()
        
        # Set content if available
        if "default" in _indexed_documents:
            engine.set_learning_content(_indexed_documents["default"])
        
        engine.start()
        
        return {
            "status": "success",
            "message": "🧠 Adaptive Learning Engine started",
            "auto_adapt": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptive/stop")
def stop_adaptive_engine():
    """Stop the adaptive learning loop."""
    try:
        engine = get_adaptive_engine()
        engine.stop()
        
        return {
            "status": "success",
            "message": "⏹️ Adaptive Learning Engine stopped"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptive/force")
def force_adaptation(request: AdaptationRequest):
    """Force adaptation to a specific state."""
    try:
        from utils.adaptive_engine import LearningState
        
        engine = get_adaptive_engine()
        
        # Parse state
        state = None
        if request.state:
            state = LearningState(request.state)
        
        engine.force_adaptation(state)
        
        return {
            "status": "success",
            "message": f"🔄 Forced adaptation to {request.state or 'current state'}",
            "current_state": engine.get_current_state().value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adaptive/strategies")
def get_adaptation_strategies():
    """Get all adaptation strategies."""
    from utils.adaptive_engine import AdaptationStrategy, LearningState
    
    strategies = {}
    for state in LearningState:
        if state != LearningState.UNKNOWN:
            strategies[state.value] = {
                "strategy": AdaptationStrategy.get_strategy(state),
                "tutor_phrases": AdaptationStrategy.TUTOR_PHRASES.get(state, [])
            }
    
    return {
        "status": "success",
        "strategies": strategies
    }


@router.get("/adaptive/history")
def get_adaptation_history():
    """Get adaptation history."""
    try:
        engine = get_adaptive_engine()
        
        return {
            "status": "success",
            "history": engine._state_history[-20:],  # Last 20 adaptations
            "total_adaptations": engine._adaptation_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

