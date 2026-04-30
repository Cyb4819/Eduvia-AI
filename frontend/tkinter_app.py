import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import time
import requests
import speech_recognition as sr
import pyttsx3
from PIL import Image, ImageTk
import io
import os
import sys

# Try to import PyMuPDF for local PDF rendering
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Backend API base URL
API_BASE = "http://127.0.0.1:8000"

# Default PDF path
DEFAULT_PDF = r"C:\Users\somos\OneDrive\Desktop\Future of Education - Eduvia AI\jesc101.pdf"


class EduviaApp:
    """Main Eduvia AI Desktop Application."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Eduvia AI - Adaptive Learning Tutor")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e2e")
        
        # State variables
        self.current_page = 1
        self.total_pages = 1
        self.pdf_zoom = 1.5
        self.learning_state = "unknown"
        self.is_voice_enabled = True
        self.is_listening = False
        self.camera_running = False
        self.backend_connected = False
        
        # PDF document (loaded locally)
        self.pdf_doc = None
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # Threading
        self.frame_poll_thread = None
        self.observer_poll_thread = None
        self.bridge_poll_thread = None
        self.thinking_poll_thread = None
        self.stop_threads = False
        
        # Thinking indicator tracking
        self._thinking_line_start = None
        self._thinking_line_end = None
        
        # Build UI
        self._build_ui()
        
        # Initialize
        self._load_pdf_local()
        self._start_camera_polling()
        self._start_state_polling()
        self._start_bridge_polling()
        self._start_thinking_polling()
        
        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _build_ui(self):
        """Build the main user interface."""
        self.main_frame = tk.Frame(self.root, bg="#1e1e2e")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self._build_status_bar()
        self._build_content_area()
        self._build_controls()
    
    def _build_status_bar(self):
        """Build top status bar showing learning state."""
        self.status_frame = tk.Frame(self.main_frame, bg="#2d2d44", height=50)
        self.status_frame.pack(fill=tk.X, pady=(0, 10))
        self.status_frame.pack_propagate(False)
        
        title = tk.Label(
            self.status_frame, 
            text="🎓 Eduvia AI - Adaptive Learning Tutor",
            font=("Helvetica", 16, "bold"),
            bg="#2d2d44", fg="#cdd6f4"
        )
        title.pack(side=tk.LEFT, padx=15, pady=5)
        
        self.state_label = tk.Label(
            self.status_frame,
            text="State: Detecting...",
            font=("Helvetica", 12),
            bg="#2d2d44", fg="#cdd6f4"
        )
        self.state_label.pack(side=tk.RIGHT, padx=15, pady=5)
        
        self.state_indicator = tk.Canvas(
            self.status_frame, width=20, height=20,
            bg="#2d2d44", highlightthickness=0
        )
        self.state_indicator.pack(side=tk.RIGHT, pady=5)
        self.state_indicator.create_oval(2, 2, 18, 18, fill="#6c7086", outline="")
    
    def _build_content_area(self):
        """Build left (PDF) and right (Camera + Chat) panels."""
        self.content_frame = tk.Frame(self.main_frame, bg="#1e1e2e")
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self._build_pdf_panel()
        self._build_right_panel()
    
    def _build_pdf_panel(self):
        """Build left panel for PDF image display."""
        self.pdf_frame = tk.LabelFrame(
            self.content_frame,
            text="📚 Learning Material",
            font=("Helvetica", 12, "bold"),
            bg="#313244", fg="#cdd6f4",
            bd=2, relief=tk.GROOVE
        )
        self.pdf_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Navigation bar
        nav_frame = tk.Frame(self.pdf_frame, bg="#313244")
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.prev_btn = tk.Button(
            nav_frame, text="◀ Prev", command=self._prev_page,
            bg="#45475a", fg="#cdd6f4", activebackground="#585b70",
            font=("Helvetica", 10), relief=tk.FLAT
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.page_label = tk.Label(
            nav_frame, text="Page 1 / 1",
            font=("Helvetica", 10), bg="#313244", fg="#cdd6f4"
        )
        self.page_label.pack(side=tk.LEFT, padx=10)
        
        self.next_btn = tk.Button(
            nav_frame, text="Next ▶", command=self._next_page,
            bg="#45475a", fg="#cdd6f4", activebackground="#585b70",
            font=("Helvetica", 10), relief=tk.FLAT
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.zoom_out_btn = tk.Button(
            nav_frame, text="🔍-", command=self._zoom_out,
            bg="#45475a", fg="#cdd6f4", activebackground="#585b70",
            font=("Helvetica", 10), relief=tk.FLAT
        )
        self.zoom_out_btn.pack(side=tk.RIGHT, padx=5)
        
        self.zoom_in_btn = tk.Button(
            nav_frame, text="🔍+", command=self._zoom_in,
            bg="#45475a", fg="#cdd6f4", activebackground="#585b70",
            font=("Helvetica", 10), relief=tk.FLAT
        )
        self.zoom_in_btn.pack(side=tk.RIGHT, padx=5)
        
        # PDF Canvas with scrollbars
        self.pdf_canvas_frame = tk.Frame(self.pdf_frame, bg="#1e1e2e")
        self.pdf_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pdf_canvas = tk.Canvas(
            self.pdf_canvas_frame, bg="#1e1e2e", highlightthickness=0
        )
        self.pdf_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.pdf_vscroll = tk.Scrollbar(
            self.pdf_canvas_frame, orient=tk.VERTICAL, command=self.pdf_canvas.yview
        )
        self.pdf_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.pdf_hscroll = tk.Scrollbar(
            self.pdf_frame, orient=tk.HORIZONTAL, command=self.pdf_canvas.xview
        )
        self.pdf_hscroll.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.pdf_canvas.configure(
            yscrollcommand=self.pdf_vscroll.set,
            xscrollcommand=self.pdf_hscroll.set
        )
        
        self.pdf_image_on_canvas = None
        self.current_pdf_image = None
    
    def _build_right_panel(self):
        """Build right panel with camera feed and chat."""
        self.right_frame = tk.Frame(self.content_frame, bg="#1e1e2e")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Camera feed frame
        self.camera_frame = tk.LabelFrame(
            self.right_frame,
            text="🎥 Camera Feed",
            font=("Helvetica", 12, "bold"),
            bg="#313244", fg="#cdd6f4",
            bd=2, relief=tk.GROOVE
        )
        self.camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.camera_label = tk.Label(
            self.camera_frame,
            bg="#1e1e2e", text="Waiting for backend camera...",
            font=("Helvetica", 11), fg="#6c7086",
            wraplength=380, justify=tk.CENTER
        )
        self.camera_label.pack(padx=5, pady=5)
        
        self.camera_status_label = tk.Label(
            self.camera_frame,
            text="📷 Backend: Not connected",
            font=("Helvetica", 9),
            bg="#313244", fg="#f38ba8"
        )
        self.camera_status_label.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Chat / Tutor Responses
        self.chat_frame = tk.LabelFrame(
            self.right_frame,
            text="💬 Tutor Responses",
            font=("Helvetica", 12, "bold"),
            bg="#313244", fg="#cdd6f4",
            bd=2, relief=tk.GROOVE
        )
        self.chat_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chat_text = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            bg="#1e1e2e", fg="#cdd6f4",
            insertbackground="#cdd6f4",
            selectbackground="#585b70",
            padx=10, pady=10,
            state=tk.DISABLED
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Chat tags
        self.chat_text.tag_configure("user", foreground="#89b4fa", font=("Helvetica", 11, "bold"))
        self.chat_text.tag_configure("tutor", foreground="#a6e3a1", font=("Helvetica", 11))
        self.chat_text.tag_configure("system", foreground="#f9e2af", font=("Helvetica", 10, "italic"))
        self.chat_text.tag_configure("error", foreground="#f38ba8", font=("Helvetica", 10))
        self.chat_text.tag_configure("thinking", foreground="#fab387", font=("Helvetica", 11, "italic"))
    
    def _build_controls(self):
        """Build bottom control panel."""
        self.controls_frame = tk.Frame(self.main_frame, bg="#2d2d44", height=60)
        self.controls_frame.pack(fill=tk.X, pady=(10, 0))
        self.controls_frame.pack_propagate(False)
        
        self.voice_btn = tk.Button(
            self.controls_frame,
            text="🎤 Ask Tutor",
            command=self._toggle_voice_input,
            bg="#89b4fa", fg="#1e1e2e",
            activebackground="#b4befe",
            font=("Helvetica", 12, "bold"),
            relief=tk.FLAT, width=15
        )
        self.voice_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.voice_status = tk.Label(
            self.controls_frame,
            text="Click to speak",
            font=("Helvetica", 11),
            bg="#2d2d44", fg="#cdd6f4"
        )
        self.voice_status.pack(side=tk.LEFT, padx=10)
        
        self.voice_toggle = tk.Button(
            self.controls_frame,
            text="🔊 Voice On",
            command=self._toggle_voice_output,
            bg="#a6e3a1", fg="#1e1e2e",
            activebackground="#b9f2c3",
            font=("Helvetica", 11),
            relief=tk.FLAT, width=12
        )
        self.voice_toggle.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.stop_btn = tk.Button(
            self.controls_frame,
            text="⏹ Stop",
            command=self._stop_speaking,
            bg="#f38ba8", fg="#1e1e2e",
            activebackground="#fab387",
            font=("Helvetica", 11),
            relief=tk.FLAT, width=10
        )
        self.stop_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.conn_status = tk.Label(
            self.controls_frame,
            text="🟡 Connecting...",
            font=("Helvetica", 10),
            bg="#2d2d44", fg="#f9e2af"
        )
        self.conn_status.pack(side=tk.RIGHT, padx=15)
    
    # ==================== LOCAL PDF RENDERING ====================
    
    def _load_pdf_local(self):
        """Load and render PDF locally using PyMuPDF. No backend needed."""
        if not PYMUPDF_AVAILABLE:
            self._show_pdf_error("PyMuPDF not installed.\nRun: pip install PyMuPDF")
            return
        
        try:
            if not os.path.exists(DEFAULT_PDF):
                self._show_pdf_error(f"PDF not found:\n{DEFAULT_PDF}")
                return
            
            self.pdf_doc = fitz.open(DEFAULT_PDF)
            self.total_pages = len(self.pdf_doc)
            self._update_page_label()
            self._render_pdf_page()
            
        except ImportError:
            self._show_pdf_error("PyMuPDF not installed.\nRun: pip install PyMuPDF")
        except Exception as e:
            self._show_pdf_error(f"PDF Error: {str(e)}")
    
    def _render_pdf_page(self):
        """Render current PDF page to canvas."""
        if self.pdf_doc is None or self.current_page < 1 or self.current_page > self.total_pages:
            return
        
        try:
            page = self.pdf_doc[self.current_page - 1]
            mat = fitz.Matrix(self.pdf_zoom, self.pdf_zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            self.current_pdf_image = ImageTk.PhotoImage(img)
            
            # Update canvas
            self.pdf_canvas.delete("all")
            self.pdf_image_on_canvas = self.pdf_canvas.create_image(
                0, 0, anchor=tk.NW, image=self.current_pdf_image
            )
            self.pdf_canvas.configure(scrollregion=self.pdf_canvas.bbox("all"))
            self._update_page_label()
            
        except Exception as e:
            self._show_pdf_error(f"Render error: {str(e)}")
    
    def _show_pdf_error(self, message):
        """Show error message on PDF canvas."""
        self.pdf_canvas.delete("all")
        self.pdf_canvas.create_text(
            200, 150, text=message,
            fill="#f38ba8", font=("Helvetica", 12),
            justify=tk.CENTER
        )
    
    def _update_page_label(self):
        """Update page counter label."""
        self.page_label.config(text=f"Page {self.current_page} / {self.total_pages}")
    
    def _prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._render_pdf_page()
    
    def _next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self._render_pdf_page()
    
    def _zoom_in(self):
        self.pdf_zoom = min(4.0, self.pdf_zoom + 0.5)
        self._render_pdf_page()
    
    def _zoom_out(self):
        self.pdf_zoom = max(1.0, self.pdf_zoom - 0.5)
        self._render_pdf_page()
    
    # ==================== CAMERA FRAME POLLING ====================
    
    def _start_camera_polling(self):
        """Start polling backend for camera frames."""
        self.camera_running = True
        self.frame_poll_thread = threading.Thread(target=self._frame_poll_loop, daemon=True)
        self.frame_poll_thread.start()
    
    def _frame_poll_loop(self):
        """Poll backend for camera frames at ~5 FPS."""
        fail_count = 0
        while self.camera_running and not self.stop_threads:
            try:
                response = requests.get(
                    f"{API_BASE}/api/camera/frame",
                    timeout=5
                )
                
                if response.status_code == 200:
                    fail_count = 0
                    image = Image.open(io.BytesIO(response.content))
                    
                    # Resize to fit panel
                    max_width = 400
                    max_height = 300
                    img_width, img_height = image.size
                    ratio = min(max_width / img_width, max_height / img_height, 1.0)
                    new_size = (int(img_width * ratio), int(img_height * ratio))
                    
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    
                    self.root.after(0, self._update_camera_frame, photo)
                else:
                    fail_count += 1
                    if fail_count >= 3:
                        self.root.after(0, self._show_camera_offline)
                        
            except Exception:
                fail_count += 1
                if fail_count >= 3:
                    self.root.after(0, self._show_camera_offline)
            
            time.sleep(0.2)  # 5 FPS
    
    def _update_camera_frame(self, photo):
        """Update camera frame display with live image."""
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo
        self.camera_status_label.config(
            text="📷 Live | Face detection active",
            fg="#a6e3a1"
        )
    
    def _show_camera_offline(self):
        """Show camera offline message."""
        self.camera_label.config(
            image="",
            text="📷 Camera feed unavailable\n\nStart backend to see live feed:\ncd backend && uvicorn main:app --reload",
            fg="#f38ba8"
        )
        self.camera_label.image = None
        self.camera_status_label.config(
            text="📷 Backend: Not connected",
            fg="#f38ba8"
        )
    
    # ==================== LEARNING STATE POLLING ====================
    
    def _start_state_polling(self):
        """Start polling backend for learning state."""
        self.observer_poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.observer_poll_thread.start()
    
    def _poll_loop(self):
        """Poll learning state from backend."""
        while not self.stop_threads:
            try:
                response = requests.get(
                    f"{API_BASE}/learning-state",
                    timeout=5
                )
                data = response.json()
                
                state = data.get("learning_state", "unknown")
                confidence = data.get("confidence", 0.0)
                
                self.root.after(0, self._update_state, state, confidence)
                self.root.after(0, self._set_connection_status, True)
                
            except Exception:
                self.root.after(0, self._set_connection_status, False)
            
            time.sleep(2)
    
    def _update_state(self, state: str, confidence: float):
        """Update learning state display ONLY in status bar."""
        self.learning_state = state
        
        self.state_label.config(text=f"State: {state.upper()} ({confidence:.0%})")
        
        colors = {
            "focused": "#a6e3a1",
            "distracted": "#f9e2af",
            "overloaded": "#f38ba8",
            "low_engagement": "#cba6f7",
            "unknown": "#6c7086"
        }
        color = colors.get(state, "#6c7086")
        
        self.state_indicator.delete("all")
        self.state_indicator.create_oval(2, 2, 18, 18, fill=color, outline="")
    
    def _set_connection_status(self, connected: bool):
        """Update connection status indicator only."""
        self.backend_connected = connected
        if connected:
            self.conn_status.config(text="🟢 Connected", fg="#a6e3a1")
        else:
            self.conn_status.config(text="🔴 Disconnected", fg="#f38ba8")
    
    # ==================== BRIDGE MESSAGE POLLING ====================
    
    def _start_bridge_polling(self):
        """Start polling backend for auto-intervention messages."""
        self.bridge_poll_thread = threading.Thread(target=self._bridge_poll_loop, daemon=True)
        self.bridge_poll_thread.start()
    
    def _bridge_poll_loop(self):
        """Poll bridge messages every 2 seconds."""
        while not self.stop_threads:
            try:
                response = requests.get(
                    f"{API_BASE}/api/bridge/messages",
                    timeout=5
                )
                data = response.json()
                messages = data.get("messages", [])
                
                for msg in messages:
                    msg_type = msg.get("type", "")
                    response_text = msg.get("response", "")
                    state = msg.get("state", "")
                    
                    if msg_type == "intervention":
                        self.root.after(0, self._add_chat_message, "system", f"🤖 Auto-intervention [{state}]")
                        self.root.after(0, self._add_chat_message, "tutor", f"🎓 {response_text}")
                    elif msg_type == "voice_response":
                        question = msg.get("question", "")
                        self.root.after(0, self._add_chat_message, "user", f"🎤 {question}")
                        self.root.after(0, self._add_chat_message, "tutor", f"🎓 {response_text}")
                    
                    # Speak if voice enabled
                    if self.is_voice_enabled and response_text:
                        self.root.after(0, self._speak_text, response_text)
                        
            except Exception:
                pass
            
            time.sleep(2)
    
    def _start_thinking_polling(self):
        """Start polling backend thinking status."""
        self.thinking_poll_thread = threading.Thread(target=self._thinking_poll_loop, daemon=True)
        self.thinking_poll_thread.start()
    
    def _thinking_poll_loop(self):
        """Poll thinking status every 1 second."""
        while not self.stop_threads:
            try:
                response = requests.get(
                    f"{API_BASE}/api/bridge/thinking",
                    timeout=5
                )
                data = response.json()
                is_generating = data.get("is_generating", False)
                
                self.root.after(0, self._update_thinking_indicator, is_generating)
                
            except Exception:
                self.root.after(0, self._update_thinking_indicator, False)
            
            time.sleep(1)
    
    def _update_thinking_indicator(self, is_generating: bool):
        """Show or hide the thinking indicator in chat."""
        if is_generating:
            # Only add if not already showing
            if self._thinking_line_start is None:
                self.chat_text.config(state=tk.NORMAL)
                self._thinking_line_start = self.chat_text.index(tk.END)
                self.chat_text.insert(tk.END, "\n🤔 Thinking...\n", "thinking")
                self._thinking_line_end = self.chat_text.index(tk.END)
                self.chat_text.see(tk.END)
                self.chat_text.config(state=tk.DISABLED)
        else:
            # Remove thinking indicator if present
            if self._thinking_line_start is not None:
                self.chat_text.config(state=tk.NORMAL)
                self.chat_text.delete(self._thinking_line_start, self._thinking_line_end)
                self.chat_text.config(state=tk.DISABLED)
                self._thinking_line_start = None
                self._thinking_line_end = None
    
    # ==================== VOICE FUNCTIONS ====================
    
    def _toggle_voice_input(self):
        """Toggle voice input (speech recognition)."""
        if self.is_listening:
            self.is_listening = False
            self.voice_btn.config(text="🎤 Ask Tutor", bg="#89b4fa")
            self.voice_status.config(text="Click to speak")
        else:
            self.is_listening = True
            self.voice_btn.config(text="⏹ Stop", bg="#f38ba8")
            self.voice_status.config(text="Listening...")
            threading.Thread(target=self._listen_for_speech, daemon=True).start()
    
    def _listen_for_speech(self):
        """Listen for speech and process."""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.root.after(0, self.voice_status.config, {"text": "Listening... (speak now)"})
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            self.root.after(0, self.voice_status.config, {"text": "Processing..."})
            transcript = self.recognizer.recognize_google(audio)
            self.root.after(0, self._process_voice_input, transcript)
            
        except sr.WaitTimeoutError:
            self.root.after(0, self.voice_status.config, {"text": "No speech detected"})
        except sr.UnknownValueError:
            self.root.after(0, self.voice_status.config, {"text": "Could not understand"})
        except Exception as e:
            self.root.after(0, self.voice_status.config, {"text": f"Error: {str(e)}"})
        finally:
            self.is_listening = False
            self.root.after(0, self.voice_btn.config, {"text": "🎤 Ask Tutor", "bg": "#89b4fa"})
    
    def _process_voice_input(self, transcript: str):
        """Process voice input and get tutor response."""
        self._add_chat_message("user", f"🎤 {transcript}")
        self.voice_status.config(text="Thinking...")
        
        threading.Thread(
            target=self._get_tutor_response,
            args=(transcript,),
            daemon=True
        ).start()
    
    def _get_tutor_response(self, question: str):
        """Get tutor response from backend."""
        try:
            response = requests.get(
                f"{API_BASE}/api/tutor/ask",
                params={"question": question, "learning_state": self.learning_state},
                timeout=60
            )
            data = response.json()
            answer = data.get("answer", "No response")
            
            self.root.after(0, self._add_chat_message, "tutor", f"🎓 {answer}")
            
            if self.is_voice_enabled:
                self.root.after(0, self._speak_text, answer)
            else:
                self.root.after(0, self.voice_status.config, {"text": "Ready"})
            
        except Exception as e:
            self.root.after(0, self._add_chat_message, "error", f"Tutor unavailable: {str(e)}")
            self.root.after(0, self.voice_status.config, {"text": "Backend not connected"})
    
    def _toggle_voice_output(self):
        """Toggle voice output (TTS)."""
        self.is_voice_enabled = not self.is_voice_enabled
        if self.is_voice_enabled:
            self.voice_toggle.config(text="🔊 Voice On", bg="#a6e3a1")
        else:
            self.voice_toggle.config(text="🔇 Voice Off", bg="#6c7086")
    
    def _speak_text(self, text: str):
        """Speak text using TTS."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.voice_status.config(text="Ready")
        except Exception as e:
            self.voice_status.config(text=f"TTS error: {str(e)}")
    
    def _stop_speaking(self):
        """Stop current speech."""
        try:
            self.tts_engine.stop()
            self.voice_status.config(text="Stopped")
        except:
            pass
    
    # ==================== CHAT FUNCTIONS ====================
    
    def _add_chat_message(self, tag: str, message: str):
        """Add a message to the chat panel."""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, f"\n{message}\n", tag)
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)
    
    # ==================== CLEANUP ====================
    
    def _on_close(self):
        """Clean up resources on window close."""
        self.stop_threads = True
        self.camera_running = False
        
        if self.pdf_doc:
            self.pdf_doc.close()
        
        self.root.destroy()


def main():
    root = tk.Tk()
    app = EduviaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

