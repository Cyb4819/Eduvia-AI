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
        observer.start()
        # Wire observer state changes to the voice bridge
        observer.set_state_change_callback(bridge.update_state)
        print("🧿 Behavioral Observer started & wired to Camera-Voice Bridge")
        
        # Preload model and index default PDF in background to avoid first-request delay
        def _preload():
            try:
                import os
                import time
                from utils.content_processor import get_content_processor
                from utils.pdf_extractor import PDFExtractor
                
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
                processor = get_content_processor(model_path)
                
                # Load model first (may take time)
                print("🔄 Loading Gemma 4 model (this may take a moment)...")
                processor.load_model()
                print("✅ Gemma 4 model preloaded")
                
                # Then index the PDF
                pdf_path = r"C:\Users\somos\OneDrive\Desktop\Future of Education - Eduvia AI\jesc101.pdf"
                if os.path.exists(pdf_path):
                    print("📚 Indexing PDF for RAG...")
                    extractor = PDFExtractor(pdf_path)
                    result = extractor.extract()
                    processor.index_document_for_rag(result['clean_text'], "default")
                    print("✅ Default PDF indexed for RAG")
                else:
                    print(f"⚠️ PDF not found at: {pdf_path}")
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
    
    DEFAULT_PDF_PATH = r"C:\Users\somos\OneDrive\Desktop\Future of Education - Eduvia AI\jesc101.pdf"
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
    
    DEFAULT_PDF_PATH = r"C:\Users\somos\OneDrive\Desktop\Future of Education - Eduvia AI\jesc101.pdf"
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


@app.post("/api/voice/ask")
async def voice_ask(request: VoiceAskRequest):
    """
    Voice-triggered Q&A endpoint.
    
    This ALWAYS processes the question regardless of learning state.
    Used when user explicitly asks via voice input.
    """
    try:
        from utils.content_processor import get_content_processor
        import os
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
        processor = get_content_processor(model_path)
        
        # Get answer with RAG
        result = processor.tutor_interaction(
            request.question,
            request.learning_state or "focused",
            "default"
        )
        
        return {
            "status": "success",
            "question": request.question,
            "answer": result.get("answer", "No response"),
            "learning_state": request.learning_state,
            "rag_enabled": result.get("rag_enabled", False),
            "sources": result.get("sources", [])
        }
        
    except Exception as e:
        return {
            "status": "error",
            "question": request.question,
            "answer": f"Error processing question: {str(e)}",
            "error": str(e)
        }


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