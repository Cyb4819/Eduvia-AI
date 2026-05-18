import sys
import os
import math
import random
import time
import requests

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QScrollArea, QFrame,
    QLineEdit, QTabWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QMessageBox, QMenu, QComboBox, QSizePolicy,
    QProgressBar, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QRectF, QTimer
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QBrush, QFont, QCursor

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


API_BASE = "http://127.0.0.1:8000"

# --- Backend Polling Threads ---

class StatePollingThread(QThread):
    state_updated = pyqtSignal(str, float)

    def run(self):
        while not self.isInterruptionRequested():
            try:
                r = requests.get(f"{API_BASE}/learning-state", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    self.state_updated.emit(
                        data.get("learning_state", "unknown"),
                        data.get("confidence", 0.0)
                    )
            except Exception:
                pass
            time.sleep(2)


class BridgePollingThread(QThread):
    intervention_received = pyqtSignal(str, str)
    voice_response_received = pyqtSignal(str)

    def run(self):
        while not self.isInterruptionRequested():
            try:
                r = requests.get(f"{API_BASE}/api/bridge/messages", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    for msg in data.get("messages", []):
                        msg_type = msg.get("type", "")
                        response_text = msg.get("response", "") or ""
                        if msg_type == "intervention":
                            state = msg.get("state", "")
                            self.intervention_received.emit(state, response_text)
                        elif msg_type == "voice_response":
                            self.voice_response_received.emit(response_text)
            except Exception:
                pass
            time.sleep(2)


class ThinkingPollingThread(QThread):
    thinking_status = pyqtSignal(bool)

    def run(self):
        while not self.isInterruptionRequested():
            try:
                r = requests.get(f"{API_BASE}/api/bridge/thinking", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    self.thinking_status.emit(data.get("is_generating", False))
                else:
                    self.thinking_status.emit(False)
            except Exception:
                self.thinking_status.emit(False)
            time.sleep(1)


class AskTutorThread(QThread):
    response_received = pyqtSignal(str)

    def __init__(self, question, learning_state):
        super().__init__()
        self.question = question
        self.learning_state = learning_state

    def run(self):
        try:
            r = requests.get(
                f"{API_BASE}/api/tutor/ask",
                params={"question": self.question, "learning_state": self.learning_state},
                timeout=310
            )
            data = r.json()
            answer = data.get("answer", data.get("message", "No response"))
            try:
                requests.post(f"{API_BASE}/api/voice/speak", json={"text": answer, "async": True}, timeout=10)
            except Exception:
                pass
        except Exception as e:
            answer = f"❌ Tutor error: {e}"
        self.response_received.emit(answer)

class PDFWorkerThread(QThread):
    progress_update = pyqtSignal(str)
    summary_ready = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, pdf_path):
        super().__init__()
        self.pdf_path = pdf_path
        
    def run(self):
        try:
            self.progress_update.emit("Indexing PDF for RAG...")
            requests.post(
                f"{API_BASE}/api/tutor/index-document", 
                json={"pdf_path": self.pdf_path, "doc_name": "default"}, 
                timeout=1200
            )
            
            self.progress_update.emit("Extracting smart summary and content...")
            r = requests.post(
                f"{API_BASE}/api/content/process-pdf?task=summarize&pdf_path={self.pdf_path}",
                timeout=1200  # Increased timeout
            )
            if r.status_code == 200:
                data = r.json()
                summary = data.get("processed_content", {}).get("summary", "")
                
                # Format summary string
                full_summary = f"📑 Smart Summary:\n\n{summary}"
                
                self.summary_ready.emit(full_summary)
            else:
                self.summary_ready.emit("Failed to generate summary.")
                
        except Exception as e:
            self.summary_ready.emit(f"Error extracting summary: {e}")
            
        self.finished.emit()


class TranslationWorker(QThread):
    result_ready = pyqtSignal(str)
    
    def __init__(self, text, lang):
        super().__init__()
        self.text = text
        self.lang = lang
        
    def run(self):
        try:
            # Query backend with prompt to translate
            prompt = f"Translate the following text to {self.lang}. Keep the exact same formatting, headers, and bullet points. Respond ONLY with the translation, no extra commentary:\n\n{self.text}"
            res = requests.post(f"{API_BASE}/api/tutor/ask", json={
                "question": prompt,
                "learning_state": "focused"
            }, timeout=300)
            if res.status_code == 200:
                ans = res.json().get("answer", "Translation failed.")
                self.result_ready.emit(ans)
            else:
                self.result_ready.emit("Error during translation request.")
        except Exception as e:
            self.result_ready.emit(f"Translation failed: {str(e)}")


# --- Custom Widgets ---

class DonutChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.state = "Confused"
        self.confidence = 0

    def update_data(self, state, confidence):
        self.state = state
        self.confidence = int(confidence * 100)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        cx = rect.width() // 2
        cy = rect.height() // 2
        ro = min(cx, cy) - 10
        ri = ro - 20

        colors = {
            "Focused": QColor("#22c55e"),
            "Distracted": QColor("#f59e0b"),
            "Overloaded": QColor("#ef4444"),
            "Confused": QColor("#3b82f6")
        }

        start_angle = 0
        span_angle = 360 * 16  

        # Draw background ring
        painter.setPen(QPen(QColor("#2a2d3e"), ro - ri))
        painter.drawArc(cx - ro + (ro - ri) // 2, cy - ro + (ro - ri) // 2, 
                        2 * ro - (ro - ri), 2 * ro - (ro - ri), 0, 360 * 16)

        # Draw active segment based on state
        active_color = colors.get(self.state, QColor("#6b7280"))
        
        if self.confidence > 0:
            span = int((self.confidence / 100) * 360 * 16)
            painter.setPen(QPen(active_color, ro - ri))
            painter.drawArc(cx - ro + (ro - ri) // 2, cy - ro + (ro - ri) // 2, 
                            2 * ro - (ro - ri), 2 * ro - (ro - ri), 90 * 16, -span)

        # Draw text
        painter.setPen(active_color if self.state == "Focused" else QColor("#e8eaf0"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(rect.adjusted(0, -15, 0, -15), Qt.AlignmentFlag.AlignCenter, self.state)

        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(rect.adjusted(0, 15, 0, 15), Qt.AlignmentFlag.AlignCenter, f"{self.confidence}%")


class PDFGraphicsView(QGraphicsView):
    text_selected = pyqtSignal(str, int, int) # text, x, y (global)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.doc = None
        self.zoom = 1.0
        
        # Rubber band selection for text
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.ContainsItemShape)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
    def load_pdf(self, doc):
        self.doc = doc
        self.render_pages()
        
    def render_pages(self):
        if not self.doc: return
        scene = QGraphicsScene(self)
        y_offset = 0
        
        for i in range(len(self.doc)):
            page = self.doc[i]
            matrix = fitz.Matrix(self.zoom * 1.5, self.zoom * 1.5)
            pix = page.get_pixmap(matrix=matrix)
            
            fmt = QImage.Format.Format_RGBA8888 if pix.alpha else QImage.Format.Format_RGB888
            qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
            
            pixmap = QPixmap.fromImage(qimg)
            item = QGraphicsPixmapItem(pixmap)
            item.setData(0, i)  # Store page index
            item.setPos(0, y_offset)
            scene.addItem(item)
            
            y_offset += pix.height + 20
            
        self.setScene(scene)
        
    def mouseReleaseEvent(self, event):
        rect = self.rubberBandRect()
        super().mouseReleaseEvent(event)
        if rect.isValid() and rect.width() > 10 and self.doc:
            scene_rect = self.mapToScene(rect).boundingRect()
            
            items = self.scene().items(scene_rect)
            if not items: return
            
            # Use the bottom-most item (which is our pixmap)
            item = items[-1]
            page_idx = item.data(0)
            
            if page_idx is not None:
                local_rect = item.mapFromScene(scene_rect).boundingRect()
                scale = self.zoom * 1.5
                pdf_rect = fitz.Rect(
                    local_rect.left() / scale,
                    local_rect.top() / scale,
                    local_rect.right() / scale,
                    local_rect.bottom() / scale
                )
                
                extracted_text = self.doc[page_idx].get_text("text", clip=pdf_rect).strip()
                if extracted_text:
                    global_pos = self.mapToGlobal(rect.bottomRight())
                    self.text_selected.emit(extracted_text, global_pos.x(), global_pos.y())
            
    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        if not self.doc or not self.scene(): return
        
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        scene_pos = self.mapToScene(pos)
        items = self.scene().items(scene_pos)
        if not items: return
        
        item = None
        for it in items:
            if isinstance(it, QGraphicsPixmapItem):
                item = it
                break
        
        if item is not None:
            page_idx = item.data(0)
            if page_idx is not None:
                local_pos = item.mapFromScene(scene_pos)
                scale = self.zoom * 1.5
                px = local_pos.x() / scale
                py = local_pos.y() / scale
                
                words = self.doc[page_idx].get_text("words")
                selected_word = None
                for w_box in words:
                    x0, y0, x1, y1, word, block_no, line_no, word_no = w_box[:8]
                    if x0 <= px <= x1 and y0 <= py <= y1:
                        # Grab the block text for a fuller context/sentence if they double click
                        # or just the word itself. Let's do the word itself first, but wait!
                        # If we can do the whole line, it's even better!
                        # Let's search for the line text that contains this word.
                        selected_word = word
                        break
                
                if selected_word:
                    global_pos = self.mapToGlobal(pos)
                    self.text_selected.emit(selected_word, global_pos.x(), global_pos.y())
            

# --- Main Window ---

class EduviaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eduvia AI")
        self.resize(1380, 900)
        self.setObjectName("mainWindow")

        self.pdf_doc = None
        self.current_pdf_path = None
        self.current_learning_state = "unknown"
        self.monitoring_active = False
        self.search_highlights = []
        self.last_selected_text = ""

        self._build_ui()
        self._load_stylesheet()
        
        # Threads
        self.state_thread = StatePollingThread()
        self.state_thread.state_updated.connect(self._on_state_updated)
        self.state_thread.start()

        self.bridge_thread = BridgePollingThread()
        self.bridge_thread.intervention_received.connect(self._on_intervention)
        self.bridge_thread.voice_response_received.connect(self._on_voice_response)
        self.bridge_thread.start()

        self.thinking_thread = ThinkingPollingThread()
        self.thinking_thread.thinking_status.connect(self._on_thinking_status)
        self.thinking_thread.start()

    def _load_stylesheet(self):
        style_path = os.path.join(os.path.dirname(__file__), "style.qss")
        if os.path.exists(style_path):
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Navbar
        navbar = QFrame()
        navbar.setObjectName("navbar")
        navbar.setFixedHeight(60)
        nav_layout = QHBoxLayout(navbar)
        
        brand = QLabel("⬡ Eduvia AI")
        brand.setObjectName("brandLabel")
        nav_layout.addWidget(brand)
        
        nav_layout.addStretch()
        
        self.btn_monitoring = QPushButton("Monitoring: OFF")
        self.btn_monitoring.setObjectName("monitoringBtn")
        self.btn_monitoring.setCheckable(True)
        self.btn_monitoring.clicked.connect(self._toggle_monitoring)
        nav_layout.addWidget(self.btn_monitoring)
        
        btn_new = QPushButton("+ New")
        btn_new.setObjectName("newBtn")
        nav_layout.addWidget(btn_new)
        
        main_layout.addWidget(navbar)

        # 2. Body
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # PDF Area
        pdf_container = QWidget()
        pdf_container.setObjectName("pdfArea")
        pdf_layout = QVBoxLayout(pdf_container)
        
        # PDF Toolbar
        pdf_toolbar = QFrame()
        pdf_toolbar.setObjectName("pdfToolbar")
        pdf_toolbar.setFixedHeight(45)
        pdf_toolbar_layout = QHBoxLayout(pdf_toolbar)
        pdf_toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        self.lbl_pdf_name = QLabel("No PDF Loaded")
        self.lbl_pdf_name.setStyleSheet("color: #e8eaf0; font-weight: bold;")
        pdf_toolbar_layout.addWidget(self.lbl_pdf_name)
        
        pdf_toolbar_layout.addStretch()
        
        self.lbl_page = QLabel("1 / 1")
        self.lbl_page.setStyleSheet("color: #a0aabf; margin-right: 15px;")
        pdf_toolbar_layout.addWidget(self.lbl_page)
        
        btn_zoom_out = QPushButton("−")
        btn_zoom_out.clicked.connect(self._zoom_out)
        pdf_toolbar_layout.addWidget(btn_zoom_out)
        
        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setStyleSheet("color: #e8eaf0;")
        pdf_toolbar_layout.addWidget(self.lbl_zoom)
        
        btn_zoom_in = QPushButton("+")
        btn_zoom_in.clicked.connect(self._zoom_in)
        pdf_toolbar_layout.addWidget(btn_zoom_in)
        
        # Add Search Bar beside zoom controls
        self.txt_pdf_search = QLineEdit()
        self.txt_pdf_search.setPlaceholderText("🔍 Search in PDF...")
        self.txt_pdf_search.setFixedWidth(180)
        self.txt_pdf_search.setStyleSheet("""
            QLineEdit {
                background-color: #1e2230;
                color: #e8eaf0;
                border: 1px solid #4f6ef7;
                border-radius: 4px;
                padding: 2px 5px;
                margin-left: 15px;
            }
        """)
        self.txt_pdf_search.returnPressed.connect(self._search_pdf_text)
        pdf_toolbar_layout.addWidget(self.txt_pdf_search)
        
        pdf_layout.addWidget(pdf_toolbar)
        
        # PDF Viewer
        self.pdf_view = PDFGraphicsView()
        self.pdf_view.text_selected.connect(self._show_text_selection_popup)
        pdf_layout.addWidget(self.pdf_view)
        
        # Upload Layout
        self.upload_widget = QWidget()
        upload_layout = QVBoxLayout(self.upload_widget)
        upload_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_upload = QPushButton("Choose PDF File")
        self.btn_upload.setObjectName("newBtn")
        self.btn_upload.setFixedSize(200, 50)
        self.btn_upload.clicked.connect(self._open_pdf)
        upload_layout.addWidget(self.btn_upload, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_upload_status = QLabel("Upload PDF then extracting content...")
        self.lbl_upload_status.setStyleSheet("color: #e8eaf0; font-size: 14px; margin-top: 10px;")
        self.lbl_upload_status.hide()
        upload_layout.addWidget(self.lbl_upload_status, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(300, 4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0) # Indeterminate mode
        self.progress_bar.setStyleSheet("""
            QProgressBar { background-color: #2a2d3e; border: none; }
            QProgressBar::chunk { background-color: #4f6ef7; }
        """)
        self.progress_bar.hide()
        upload_layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Add upload widget on top of pdf view logic
        self.upload_overlay = self.upload_widget
        pdf_layout.addWidget(self.upload_overlay, stretch=1)
        self.pdf_view.hide()
        
        body_layout.addWidget(pdf_container, stretch=2)

        # Right Panel
        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")
        right_panel.setFixedWidth(380)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.setObjectName("tabWidget")
        
        # Summary Tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        summ_header = QHBoxLayout()
        summ_header.addWidget(QLabel("Summary", objectName="summaryTitle"))
        summ_header.addStretch()
        
        self.btn_summarize = QComboBox()
        self.btn_summarize.setObjectName("summarizeDropdown")
        self.btn_summarize.addItems(["Summarize", "Download", "Translate"])
        self.btn_summarize.activated.connect(self._handle_summary_action)
        summ_header.addWidget(self.btn_summarize)
        
        summary_layout.addLayout(summ_header)
        
        self.summ_box = QLabel("Waiting for PDF to process...")
        self.summ_box.setObjectName("summaryBox")
        self.summ_box.setWordWrap(True)
        self.summ_box.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        summ_scroll = QScrollArea()
        summ_scroll.setWidgetResizable(True)
        summ_scroll.setWidget(self.summ_box)
        summ_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        summary_layout.addWidget(summ_scroll, stretch=1)
        self.tabs.addTab(summary_tab, "Summary")
        
        # Stats Tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        stats_layout.addWidget(QLabel("Live Cognitive State", objectName="summaryTitle"))
        self.donut = DonutChart()
        stats_layout.addWidget(self.donut, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_stats_details = QLabel("Focused: 0%\nDistracted: 0%\nOverloaded: 0%\nConfused: 0%")
        self.lbl_stats_details.setStyleSheet("color: #a0aabf; font-size: 14px; line-height: 1.5;")
        stats_layout.addWidget(self.lbl_stats_details)
        stats_layout.addStretch()
        
        self.tabs.addTab(stats_tab, "Stats")
        
        # AI Chat Tab
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        chat_layout.setContentsMargins(0, 10, 0, 0)
        
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chatScrollArea")
        self.chat_scroll.setWidgetResizable(True)
        
        self.chat_content = QWidget()
        self.chat_content_layout = QVBoxLayout(self.chat_content)
        self.chat_content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_scroll.setWidget(self.chat_content)
        
        chat_layout.addWidget(self.chat_scroll)
        
        self.lbl_generating = QLabel("🤔 GENERATING RESPONSE...")
        self.lbl_generating.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 14px; padding: 10px; background-color: #21253a; border-radius: 8px; margin: 0px 10px;")
        self.lbl_generating.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_generating.hide()
        chat_layout.addWidget(self.lbl_generating)
        
        chat_input_container = QFrame()
        chat_input_container.setObjectName("chatInputContainer")
        chat_input_layout = QHBoxLayout(chat_input_container)
        chat_input_layout.setContentsMargins(10, 5, 5, 5)
        
        self.chat_input = QLineEdit()
        self.chat_input.setObjectName("chatInput")
        self.chat_input.setPlaceholderText("Ask anything...")
        self.chat_input.returnPressed.connect(self._send_message)
        chat_input_layout.addWidget(self.chat_input)
        
        btn_send = QPushButton("➤")
        btn_send.setObjectName("chatSendBtn")
        btn_send.clicked.connect(self._send_message)
        chat_input_layout.addWidget(btn_send)
        
        chat_layout.addWidget(chat_input_container)
        
        self.tabs.addTab(chat_tab, "AI Chat")
        self.tabs.setCurrentIndex(2) # Default to chat
        
        right_layout.addWidget(self.tabs)
        body_layout.addWidget(right_panel)
        main_layout.addWidget(body)

        # Text Selection Popup Menu
        self.popup_menu = QFrame(self)
        self.popup_menu.setObjectName("textSelectionPopup")
        self.popup_menu.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.popup_menu.hide()
        
        popup_layout = QHBoxLayout(self.popup_menu)
        popup_layout.setContentsMargins(5, 5, 5, 5)
        popup_layout.setSpacing(2)
        
        for action in ["AI Chat", "Explain", "Translate"]:
            btn = QPushButton(action)
            btn.clicked.connect(lambda checked, a=action: self._handle_popup_action(a))
            popup_layout.addWidget(btn)

    def _toggle_monitoring(self):
        self.monitoring_active = self.btn_monitoring.isChecked()
        if self.monitoring_active:
            self.btn_monitoring.setText("Monitoring: ON")
            try:
                requests.post(f"{API_BASE}/api/trigger/config", json={"auto_intervention_enabled": True}, timeout=2)
                requests.post(f"{API_BASE}/api/observer/start", timeout=2)
                requests.post(f"{API_BASE}/api/adaptive/start", timeout=2)
            except: pass
        else:
            self.btn_monitoring.setText("Monitoring: OFF")
            try:
                requests.post(f"{API_BASE}/api/trigger/config", json={"auto_intervention_enabled": False}, timeout=2)
                requests.post(f"{API_BASE}/api/observer/stop", timeout=2)
                requests.post(f"{API_BASE}/api/adaptive/stop", timeout=2)
            except: pass

    def _open_pdf(self):
        if not PDF_SUPPORT:
            QMessageBox.warning(self, "Missing Dependency", "Please install PyMuPDF (fitz) first.")
            return
            
        path, _ = QFileDialog.getOpenFileName(self, "Choose PDF", "", "PDF Files (*.pdf)")
        if path:
            self.current_pdf_path = path
            self._clear_search_highlights()
            self.pdf_doc = fitz.open(path)
            self.lbl_pdf_name.setText(os.path.basename(path))
            
            # Hide upload button, show loader
            self.btn_upload.hide()
            self.lbl_upload_status.show()
            self.progress_bar.show()
            
            # Switch to summary tab while loading
            self.tabs.setCurrentIndex(0)
            self.summ_box.setText("Extracting content and generating summary...")
            
            # Background indexing & summary
            self.pdf_worker = PDFWorkerThread(path)
            self.pdf_worker.progress_update.connect(lambda msg: self.lbl_upload_status.setText(msg))
            self.pdf_worker.summary_ready.connect(self._on_summary_ready)
            self.pdf_worker.finished.connect(self._on_pdf_processed)
            self.pdf_worker.start()

    def _on_summary_ready(self, summary_text):
        self.summ_box.setText(summary_text)

    def _on_pdf_processed(self):
        self.upload_overlay.hide()
        self.pdf_view.show()
        self.pdf_view.load_pdf(self.pdf_doc)
        self._update_pdf_labels()
        
        # Automatically toggle monitoring ON now that PDF is fully loaded and summarized
        if not self.monitoring_active:
            self.btn_monitoring.setChecked(True)
            self._toggle_monitoring()

    def _zoom_in(self):
        if self.pdf_view.zoom < 2.5:
            self.pdf_view.zoom += 0.2
            self.pdf_view.render_pages()
            self._update_pdf_labels()

    def _zoom_out(self):
        if self.pdf_view.zoom > 0.5:
            self.pdf_view.zoom -= 0.2
            self.pdf_view.render_pages()
            self._update_pdf_labels()

    def _update_pdf_labels(self):
        self.lbl_zoom.setText(f"{int(self.pdf_view.zoom * 100)}%")
        if self.pdf_doc:
            self.lbl_page.setText(f"1 / {len(self.pdf_doc)}") # Simplified

    def _handle_summary_action(self, index):
        if index == 0:  # Summarize (re-trigger)
            if self.current_pdf_path:
                self.summ_box.setText("Generating summary...")
                # Run background PDF summary worker
                self.pdf_worker = PDFWorkerThread(self.current_pdf_path)
                self.pdf_worker.progress_update.connect(lambda msg: self.lbl_upload_status.setText(msg))
                self.pdf_worker.summary_ready.connect(self._on_summary_ready)
                self.pdf_worker.finished.connect(self._on_pdf_processed)
                self.pdf_worker.start()
            else:
                QMessageBox.warning(self, "Summarize PDF", "No PDF loaded to summarize.")
        elif index == 1:  # Download
            text = self.summ_box.text().strip()
            if not text or text == "Waiting for PDF to process..." or text == "Extracting content and generating summary...":
                QMessageBox.warning(self, "Download Summary", "No summary content available to download.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save Summary", "summary.txt", "Text Files (*.txt)")
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(text)
                    QMessageBox.information(self, "Download Summary", "Summary successfully saved!")
                except Exception as e:
                    QMessageBox.critical(self, "Download Summary", f"Failed to save summary: {e}")
        elif index == 2:  # Translate
            text = self.summ_box.text().strip()
            if not text or text == "Waiting for PDF to process..." or text == "Extracting content and generating summary...":
                QMessageBox.warning(self, "Translate Summary", "No summary content available to translate.")
                return
            lang, ok = QInputDialog.getItem(self, "Translate Summary", "Select Language:", ["Spanish", "French", "Hindi", "German", "Chinese", "Other"], 0, False)
            if ok and lang:
                self.summ_box.setText(f"Translating summary to {lang}...")
                self.translation_worker = TranslationWorker(text, lang)
                self.translation_worker.result_ready.connect(self._on_summary_translated)
                self.translation_worker.start()

    def _on_summary_translated(self, translated_text):
        self.summ_box.setText(translated_text)

    def _clear_search_highlights(self):
        if hasattr(self, "search_highlights"):
            for h in self.search_highlights:
                try:
                    self.pdf_view.scene().removeItem(h)
                except: pass
            self.search_highlights.clear()
        else:
            self.search_highlights = []

    def _search_pdf_text(self):
        query = self.txt_pdf_search.text().strip()
        if not query or not self.pdf_doc: return
        
        self._clear_search_highlights()
        found = False
        
        for page_idx in range(len(self.pdf_doc)):
            page = self.pdf_doc[page_idx]
            rects = page.search_for(query)
            if rects:
                items = self.pdf_view.scene().items()
                for item in items:
                    if isinstance(item, QGraphicsPixmapItem) and item.data(0) == page_idx:
                        self.pdf_view.centerOn(item)
                        scale = self.pdf_view.zoom * 1.5
                        
                        # Add highlights for all matches on this page
                        for match_rect in rects:
                            scene_rect = QRectF(
                                item.x() + match_rect.x0 * scale,
                                item.y() + match_rect.y0 * scale,
                                match_rect.width * scale,
                                match_rect.height * scale
                            )
                            highlight = self.pdf_view.scene().addRect(
                                scene_rect,
                                QPen(Qt.PenStyle.NoPen),
                                QBrush(QColor(255, 255, 0, 120)) # Semi-transparent yellow
                            )
                            self.search_highlights.append(highlight)
                        
                        found = True
                        break
                if found:
                    break
                    
        if not found:
            QMessageBox.information(self, "Search", f"No matches found for '{query}'")

    def _show_text_selection_popup(self, text, x, y):
        self.last_selected_text = text
        self.popup_menu.move(x - 50, y - 50)
        self.popup_menu.show()
        QTimer.singleShot(5000, self.popup_menu.hide)

    def _handle_popup_action(self, action):
        self.popup_menu.hide()
        self.tabs.setCurrentIndex(2)
        if action == "AI Chat":
            self.chat_input.setText(f"Explain this text from the PDF: '{self.last_selected_text}' ")
            self.chat_input.setFocus()
        elif action == "Explain":
            self._send_user_message(f"Explain this concept simply: '{self.last_selected_text}'")
        elif action == "Translate":
            lang, ok = QInputDialog.getItem(self, "Translate", "Select Language:", ["Spanish", "French", "Hindi", "German", "Chinese", "Other"], 0, False)
            if ok and lang:
                self._send_user_message(f"Translate the following text to {lang}: '{self.last_selected_text}'")

    # --- Chat Logic ---

    def _add_chat_bubble(self, text, is_user=False):
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bubble.setObjectName("userBubble" if is_user else "tutorBubble")
        bubble.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)
        
        row = QHBoxLayout()
        if is_user:
            row.addStretch()
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch()
            
        layout.addLayout(row)
        
        # Copy / Speak icons for tutor
        if not is_user:
            actions_row = QHBoxLayout()
            actions_row.setSpacing(10)
            
            btn_copy = QPushButton("📋")
            btn_copy.setStyleSheet("background: transparent; border: none; color: #a0aabf;")
            btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
            
            btn_speak = QPushButton("🔊")
            btn_speak.setStyleSheet("background: transparent; border: none; color: #a0aabf;")
            btn_speak.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_speak.clicked.connect(lambda: self._speak_text(text))
            
            actions_row.addWidget(btn_copy)
            actions_row.addWidget(btn_speak)
            actions_row.addStretch()
            layout.addLayout(actions_row)
            
        self.chat_content_layout.addWidget(container)
        
        QTimer.singleShot(100, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))

    def _speak_text(self, text):
        try:
            requests.post(f"{API_BASE}/api/voice/speak", json={"text": text, "async": True}, timeout=2)
        except: pass

    def _send_message(self):
        text = self.chat_input.text().strip()
        if not text: return
        
        self.chat_input.clear()
        self._send_user_message(text)
        
    def _send_user_message(self, text):
        self._add_chat_bubble(text, is_user=True)
        self.lbl_generating.show()
        
        self.ask_thread = AskTutorThread(text, self.current_learning_state)
        self.ask_thread.response_received.connect(self._on_tutor_response)
        self.ask_thread.start()

    def _on_tutor_response(self, text):
        self.lbl_generating.hide()
        self._add_chat_bubble(text, is_user=False)

    # --- Polling Callbacks ---

    def _on_state_updated(self, state, confidence):
        self.current_learning_state = state
        
        label_map = {
            "focused": "Focused",
            "distracted": "Distracted",
            "overloaded": "Overloaded",
            "low_engagement": "Confused",
            "unknown": "Confused",
        }
        active_label = label_map.get(state, "Confused")
        
        self.donut.update_data(active_label, confidence)
        
        pct = int(confidence * 100)
        details = (
            f"Focused: {pct}%" if active_label == "Focused" else "Focused: 0%",
            f"Distracted: {pct}%" if active_label == "Distracted" else "Distracted: 0%",
            f"Overloaded: {pct}%" if active_label == "Overloaded" else "Overloaded: 0%",
            f"Confused: {pct}%" if active_label == "Confused" else "Confused: 0%"
        )
        self.lbl_stats_details.setText("\n".join(details))

    def _on_intervention(self, state, text):
        self._speak_text(text)

    def _on_voice_response(self, text):
        self._add_chat_bubble(text, is_user=False)

    def _on_thinking_status(self, is_generating):
        if is_generating:
            self.lbl_generating.show()
        else:
            self.lbl_generating.hide()

    def closeEvent(self, event):
        self.state_thread.requestInterruption()
        self.bridge_thread.requestInterruption()
        self.thinking_thread.requestInterruption()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EduviaApp()
    window.show()
    sys.exit(app.exec())