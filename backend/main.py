import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import BackgroundTasks, Query
from routes import tutor
from utils.observer import BehavioralObserver
from utils.camera_voice_bridge import get_camera_voice_bridge
from utils.voice_trigger_controller import get_trigger_controller
import threading
import uvicorn
import atexit
import asyncio
import json
import io

app = FastAPI(
    title="Eduvia AI - Phases 1 & 3",
    description="Phase 1: Behavioral Observer + Phase 3: Learning Content Processing with Gemma 4",
    version="1.3.0"
)

observer = BehavioralObserver()
bridge = get_camera_voice_bridge()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tutor.router, prefix="/api", tags=["Tutor"])

@app.on_event("startup")
async def startup_event():
    try:
        # DO NOT start observer/engine on startup to save CPU and honor user request
        # observer.start()

        # Wire observer state changes to both:
        # 1) voice bridge (auto-intervention / TTS)
        # 2) adaptive learning engine (Phase 5 proactive teaching)
        from routes.tutor import get_adaptive_engine
        engine = get_adaptive_engine()
        # engine.start()

        def _on_state_change(state: str, confidence: float, signals: dict):
            bridge.update_state(state, confidence, signals)
            try:
                engine.update_observer_state(signals)
            except Exception as _e:
                # Don’t break observer callback if adaptive engine fails
                print(f"⚠️ Adaptive engine update failed: {_e}")

        observer.set_state_change_callback(_on_state_change)
        print("🧿 Behavioral Observer callbacks wired to Camera-Voice Bridge + Adaptive Engine (Idle until explicitly started)")

        
        # Preload model and index default PDF in background to avoid first-request delay
        def _preload():
            try:
                import os
                import time
                from utils.content_processor_fixed import get_content_processor
                from utils.pdf_extractor import PDFExtractor
                
                # Get relative paths
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
                
                # Try to find PDF - check multiple locations
                pdf_candidates = [
                    os.path.join(base_dir, "..", "jesc101.pdf"),  # Root of project
                    os.path.join(base_dir, "jesc101.pdf"),  # Backend dir
                    r"C:\Users\somos\OneDrive\Desktop\Future of Education - Eduvia AI\jesc101.pdf",  # Fallback
                ]
                
                pdf_path = None
                for candidate in pdf_candidates:
                    if os.path.exists(candidate):
                        pdf_path = candidate
                        break
                
                # Get processor (don't auto-load model in singleton getter)
                processor = get_content_processor(model_path)
                
                # Load model only if needed
                if not processor._is_loaded:
                    print("🔄 Loading Gemma 4 model (this may take a moment)...")
                    processor.load_model()
                    print("✅ Gemma 4 model preloaded")
                
                # Then index the PDF if found
                if pdf_path and os.path.exists(pdf_path):
                    print(f"📚 Indexing PDF from: {pdf_path}")
                    extractor = PDFExtractor(pdf_path)
                    result = extractor.extract()
                    processor.index_document_for_rag(result['clean_text'], "default")
                    print("✅ Default PDF indexed for RAG")
                else:
                    print(f"⚠️ PDF not found in any location. RAG will use model's general knowledge.")
            except Exception as e:
                print(f"⚠️ Preload warning: {e}")
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=_preload, daemon=True).start()
        
    except Exception as e:
        print(f"❌ Observer start failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    observer.stop()

@app.get("/signals")
async def get_signals():
    """Get latest signals (poll endpoint)"""
    return observer.get_signals()

from fastapi.responses import StreamingResponse
import time as time_mod

@app.get("/signals/stream")
async def signals_stream():
    async def event_stream():
        while True:
            signals = observer.get_signals()
            yield f"data: {json.dumps(signals)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/learning-state")
async def get_learning_state():
    """Get current learning state with confidence score"""
    signals = observer.get_signals()
    return {
        "learning_state": signals.get('learning_state', 'unknown'),
        "confidence": signals.get('learning_confidence', 0.0),
        "timestamp": signals.get('timestamp')
    }

@app.get("/learning-state/summary")
async def get_learning_state_summary():
    """Get detailed learning state analysis"""
    return observer.get_learning_state_summary()

# ============================================
# NEW: Camera Frame Endpoint
# ============================================

@app.get("/api/camera/frame")
async def camera_frame():
    """Get the latest processed camera frame with face detection overlay."""
    frame_bytes = observer.get_current_frame()
    if frame_bytes is None:
        return {"status": "error", "message": "No frame available"}
    
    return StreamingResponse(
        io.BytesIO(frame_bytes),
        media_type="image/jpeg"
    )

# ============================================
# NEW: PDF Page Image Endpoint
# ============================================

@app.get("/api/content/pdf-page-image/{page_num}")
async def get_pdf_page_image(
    page_num: int,
    pdf_path: str = Query(None, description="Path to PDF file"),
    zoom: float = Query(2.0, description="Zoom factor for rendering (default 2.0)")
):
    """
    Render a specific PDF page as a PNG image.
    
    Args:
        page_num: Page number to render (1-indexed)
        pdf_path: Path to PDF file (uses default if not provided)
        zoom: Zoom factor for rendering quality
    """
    import fitz  # PyMuPDF
    import os
    
    # Find default PDF
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_pdf_candidates = [
        os.path.join(base_dir, "..", "jesc101.pdf"),
        os.path.join(base_dir, "jesc101.pdf"),
        r"C:\Users\somos\OneDrive\Desktop\Future of Education - Eduvia AI\jesc101.pdf",
    ]
    DEFAULT_PDF_PATH = next((p for p in default_pdf_candidates if os.path.exists(p)), default_pdf_candidates[0])
    target_path = pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        return {"status": "error", "message": f"PDF not found: {target_path}"}
    
    try:
        doc = fitz.open(target_path)
        
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return {"status": "error", "message": f"Page {page_num} not found. PDF has {len(doc)} pages."}
        
        # Render page to image
        page = doc[page_num - 1]  # 0-indexed in PyMuPDF
        mat = fitz.Matrix(zoom, zoom)  # Zoom matrix for quality
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        doc.close()
        
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/png"
        )
        
    except Exception as e:
        return {"status": "error", "message": f"PDF rendering failed: {str(e)}"}

@app.get("/api/content/pdf-page-count")
async def get_pdf_page_count(
    pdf_path: str = Query(None, description="Path to PDF file")
):
    """Get total page count of a PDF."""
    import fitz
    import os
    
    # Find default PDF
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_pdf_candidates = [
        os.path.join(base_dir, "..", "jesc101.pdf"),
        os.path.join(base_dir, "jesc101.pdf"),
        r"C:\Users\somos\OneDrive\Desktop\Future of Education - Eduvia AI\jesc101.pdf",
    ]
    DEFAULT_PDF_PATH = next((p for p in default_pdf_candidates if os.path.exists(p)), default_pdf_candidates[0])
    target_path = pdf_path or DEFAULT_PDF_PATH
    
    if not os.path.exists(target_path):
        return {"status": "error", "message": f"PDF not found: {target_path}"}
    
    try:
        doc = fitz.open(target_path)
        count = len(doc)
        doc.close()
        return {"status": "success", "page_count": count, "pdf_path": target_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Eduvia AI - Phase 1 Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .card { background: white; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .status { font-size: 24px; font-weight: bold; }
        .focused { color: #4CAF50; background: #E8F5E9; padding: 10px; border-radius: 5px; }
        .distracted { color: #FF9800; background: #FFF3E0; padding: 10px; border-radius: 5px; }
        .overloaded { color: #E91E63; background: #FCE4EC; padding: 10px; border-radius: 5px; }
        .low_engagement { color: #f44336; background: #FFEBEE; padding: 10px; border-radius: 5px; }
        .signal { display: inline-block; margin: 5px; padding: 10px; background: #e3f2fd; border-radius: 5px; }
        .learning-state-box { font-size: 32px; font-weight: bold; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0; }
        #log { height: 200px; overflow-y: scroll; background: #fafafa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; }
        button { background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1976D2; }
        .confidence { font-size: 14px; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>🧿 Eduvia AI - Behavioral & Learning State Monitor</h1>
    
    <div class="card">
        <h2>📊 Current Learning State</h2>
        <div id="learning-state-display" class="learning-state-box focused">
            -- 
        </div>
        <div class="confidence">
            Confidence: <span id="confidence">--</span>
        </div>
    </div>
    
    <div class="card">
        <h2>📡 Live Behavioral Signals</h2>
        <div id="signals">
            <div class="signal">presence: <span data-presence>--</span></div>
            <div class="signal">gaze: <span data-gaze>--</span></div>
            <div class="signal">movement: <span data-movement>--</span></div>
            <div class="signal" id="attention">--</div>
        </div>
        <div style="margin-top: 20px;">
            <button onclick="toggleStream()">Start/Stop Stream</button>
            <span id="status">Connecting...</span>
        </div>
    </div>
    
    <div class="card">
        <h2>📝 Signal Log</h2>
        <div id="log"></div>
    </div>
    
    <script>
        let eventSource = null;
        let streaming = false;
        
        function updateSignals(data) {
            document.querySelector('[data-presence]').textContent = data.presence;
            document.querySelector('[data-gaze]').textContent = data.gaze_on_screen;
            document.querySelector('[data-movement]').textContent = data.movement_level;
            
            const attEl = document.getElementById('attention');
            attEl.textContent = data.attention_state;
            attEl.className = `signal ${data.attention_state}`;
            
            // Update learning state display
            const learnStateEl = document.getElementById('learning-state-display');
            const stateDisplay = {
                'focused': '✅ FOCUSED',
                'distracted': '⚠️ DISTRACTED',
                'overloaded': '🔴 OVERLOADED',
                'low_engagement': '❌ LOW ENGAGEMENT'
            };
            
            learnStateEl.textContent = stateDisplay[data.learning_state] || 'UNKNOWN';
            learnStateEl.className = `learning-state-box ${data.learning_state}`;
            document.getElementById('confidence').textContent = (data.learning_confidence * 100).toFixed(0) + '%';
        }
        
        function addToLog(data) {
            const log = document.getElementById('log');
            const entry = document.createElement('div');
            const time = new Date(data.timestamp).toLocaleTimeString();
            entry.textContent = `[${time}] State: ${data.learning_state} (${(data.learning_confidence * 100).toFixed(0)}%) | Attention: ${data.attention_state}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
            if (log.children.length > 20) log.removeChild(log.firstChild);
        }
        
        function toggleStream() {
            if (streaming) {
                if (eventSource) eventSource.close();
                streaming = false;
                document.getElementById('status').textContent = 'Stopped';
            } else {
                eventSource = new EventSource('/signals/stream');
                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateSignals(data);
                    addToLog(data);
                };
                eventSource.onerror = function() {
                    document.getElementById('status').textContent = 'Error connecting';
                };
                streaming = true;
                document.getElementById('status').textContent = 'Streaming...';
            }
        }
        
        // Auto start streaming
        toggleStream();


        setInterval(() => {
        fetch('/api/bridge/messages')
        .then(r => r.json())
        .then(data => {
            data.messages.forEach(msg => {
                if (msg.type === 'intervention') {
                    addToLog({learning_state: msg.state, timestamp: new Date().toISOString()});
                // Optional: Show popup/chat message
                    alert(`🤖 Auto-help: ${msg.response}`);
                }
            });
        });
    }, 2000);        
    </script>
</body>
</html>
    """

@app.get("/api/health")
def health():
    return {
        "status": "✅ Phase 1 Observer Integrated",
        "cam_running": observer.running,
        "latest_signals": observer.get_signals()
    }


# ============================================
# NEW: Voice & Camera Integration Endpoints
# ============================================

from pydantic import BaseModel
from typing import Optional

class VoiceAskRequest(BaseModel):
    """Request for voice-triggered Q&A."""
    question: str
    learning_state: Optional[str] = "focused"

class TriggerConfigRequest(BaseModel):
    """Request to update trigger configuration."""
    auto_intervention_enabled: Optional[bool] = None
    intervention_cooldown: Optional[float] = None
    min_confidence: Optional[float] = None


@app.get("/api/camera/status")
async def camera_status():
    """Get camera and observer status."""
    signals = observer.get_signals()
    return {
        "camera_running": observer.running,
        "signals": signals,
        "learning_state": signals.get('learning_state', 'unknown'),
        "confidence": signals.get('learning_confidence', 0.0)
    }


@app.get("/api/trigger/config")
async def get_trigger_config():
    """Get current trigger controller configuration."""
    controller = get_trigger_controller()
    return controller.get_status()


@app.post("/api/trigger/config")
async def update_trigger_config(request: TriggerConfigRequest):
    """Update trigger controller configuration."""
    controller = get_trigger_controller()
    
    if request.auto_intervention_enabled is not None:
        controller.config.auto_intervention_enabled = request.auto_intervention_enabled
    
    if request.intervention_cooldown is not None:
        controller.config.intervention_cooldown = request.intervention_cooldown
    
    if request.min_confidence is not None:
        controller.config.min_confidence = request.min_confidence
    
    return {
        "status": "success",
        "message": "Trigger configuration updated",
        "config": controller.get_status()
    }


@app.post("/api/observer/start")
async def start_observer():
    try:
        observer.start()
        return {"status": "success", "message": "Observer started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/observer/stop")
async def stop_observer():
    try:
        observer.stop()
        return {"status": "success", "message": "Observer stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================
# Voice I/O Endpoints - Complete Voice Flow
# ============================================

@app.post("/api/voice/ask")
async def voice_ask(request: VoiceAskRequest):
    """
    Voice-triggered Q&A endpoint.
    
    This ALWAYS processes the question regardless of learning state.
    Used when user explicitly asks via voice input.
    
    Returns:
        - answer: The tutor's response (generated from PDF via RAG)
        - rag_enabled: Whether PDF-based retrieval was used
        - sources: Relevant excerpts from the PDF
    """
    try:
        print(f"🎤 USER QUESTION: {request.question}")
        from utils.content_processor_fixed import get_content_processor
        import os
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
        processor = get_content_processor(model_path)
        
        # Ensure model is loaded
        if not processor._is_loaded:
            print("🔄 Loading Gemma model before response...")
            processor.load_model()
        
        # Get answer with RAG - will use PDF if indexed
        result = processor.tutor_interaction(
            request.question,
            request.learning_state or "focused",
            "default"
        )
        
        # Log response type
        if result.get("rag_enabled") and result.get("context_used"):
            print(f"✅ Response generated FROM PDF")
        else:
            print(f"⚠️ Response generated from model knowledge (PDF not indexed)")
        
        return {
            "status": "success",
            "question": request.question,
            "answer": result.get("answer", "No response"),
            "learning_state": request.learning_state,
            "rag_enabled": result.get("rag_enabled", False),
            "context_used": result.get("context_used", False),
            "sources": result.get("sources", [])
        }
        
    except Exception as e:
        print(f"❌ Voice ask failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "question": request.question,
            "answer": f"Error: {str(e)}",
            "error": str(e)
        }


@app.post("/api/voice/speak")
async def voice_speak(request: dict):
    """
    Text-to-Speech endpoint.
    
    Speaks the given text using pyttsx3.
    
    Request:
        text: Text to speak
        async: If true, speaks in background thread
    
    Response:
        status: success/error
        message: Status message
    """
    try:
        text = request.get("text", "")
        is_async = request.get("async", False)
        
        if not text:
            return {"status": "error", "message": "No text provided"}
        
        print(f"🔊 Speaking: {text[:100]}...")
        
        from utils.tts_engine import get_tts
        tts = get_tts()
        
        # Ensure TTS is initialized
        if tts.engine is None:
            tts.initialize()
        
        if is_async:
            tts.speak_async(text, word_by_word=False)
            return {"status": "success", "message": "Speaking in background"}
        else:
            tts.speak(text, word_by_word=False)
            return {"status": "success", "message": "Speech completed"}
        
    except Exception as e:
        print(f"❌ TTS failed: {e}")
        return {"status": "error", "message": str(e), "error": str(e)}


@app.post("/api/voice/full-interaction")
async def voice_full_interaction(request: dict):
    """
    Complete voice interaction: process question and speak response.
    
    This is the MAIN endpoint for voice interaction flow.
    
    Request:
        question: User's question
        learning_state: Current learning state
        speak: Whether to speak the response (default: True)
    
    Response:
        question: Echo of the question
        answer: Generated response
        spoken: Whether response was spoken
        rag_enabled: Whether PDF was used
    """
    try:
        question = request.get("question", "")
        learning_state = request.get("learning_state", "focused")
        should_speak = request.get("speak", True)
        
        if not question:
            return {"status": "error", "message": "No question provided"}
        
        print(f"🎤→🤖→🔊 FULL VOICE INTERACTION: {question}")
        
        # 1. Process question with Gemma + RAG
        from utils.content_processor_fixed import get_content_processor
        import os
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
        processor = get_content_processor(model_path)
        
        if not processor._is_loaded:
            processor.load_model()
        
        # Get answer
        result = processor.tutor_interaction(question, learning_state, "default")
        answer = result.get("answer", "")
        
        # 2. Speak if requested
        spoken = False
        if should_speak and answer:
            try:
                from utils.tts_engine import get_tts
                tts = get_tts()
                if tts.engine is None:
                    tts.initialize()
                
                print(f"🔊 Speaking response...")
                # Speak in background so API returns immediately
                tts.speak_async(answer, word_by_word=False)
                spoken = True
            except Exception as e:
                print(f"⚠️ TTS error (continuing anyway): {e}")
                # Don't fail - return answer even if TTS fails
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "learning_state": learning_state,
            "spoken": spoken,
            "rag_enabled": result.get("rag_enabled", False),
            "context_used": result.get("context_used", False),
            "sources": result.get("sources", [])
        }
        
    except Exception as e:
        print(f"❌ Full interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/api/voice/status")
async def voice_status():
    """Get current voice system status."""
    try:
        from utils.tts_engine import get_tts
        tts = get_tts()
        
        return {
            "status": "success",
            "tts_initialized": tts.engine is not None,
            "tts_speaking": tts.is_speaking(),
            "message": "✅ Voice system ready"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================
# Message Polling - For Frontend Updates
# ============================================

@app.get("/api/messages/poll")
async def poll_messages():
    """
    Poll for messages from the voice bridge (for frontend polling).
    
    This allows the frontend to retrieve messages that were generated
    by the auto-intervention system or voice responses.
    """
    messages = bridge.pop_messages()
    return {
        "status": "success",
        "messages": messages,
        "count": len(messages)
    }


# ============================================
# Auto-Intervention Endpoints
# ============================================

@app.get("/api/intervention/status")
async def intervention_status():
    """Get status of auto-intervention system."""
    controller = get_trigger_controller()
    status = controller.get_status()
    
    return {
        "auto_intervention_enabled": status.get("auto_intervention_enabled", False),
        "intervention_cooldown": status.get("intervention_cooldown", 0),
        "min_confidence": status.get("min_confidence", 0),
        "message": "Auto-intervention triggers voice responses when learner appears stuck/distracted"
    }
    


@app.get("/api/bridge/status")
async def bridge_status():
    """Get camera-voice bridge status."""
    return bridge.get_status()


@app.post("/api/bridge/stop-speaking")
async def stop_speaking():
    """Stop current voice output."""
    bridge.stop_speaking()
    return {"status": "success", "message": "Speech stopped"}


@app.get("/api/bridge/messages")
async def bridge_messages():
    """
    Poll pending auto-intervention / voice-response messages.
    Frontend should call this every 1-2 seconds.
    """
    return {
        "status": "success",
        "messages": bridge.pop_messages()
    }


@app.get("/api/bridge/thinking")
async def bridge_thinking():
    """
    Check if the backend is currently generating a tutor response.
    Frontend uses this to show/hide 'Thinking...' indicator.
    """
    return {
        "status": "success",
        "is_generating": bridge.is_generating(),
        "is_speaking": bridge.is_speaking()
    }