# Problem Statement

Traditional learning systems are static and reactive. They assume that learners can consistently maintain attention, process information at a uniform pace, and explicitly ask for help when needed.

In reality, learning is dynamic. Attention fluctuates, cognitive overload happens, and many learners—especially neurodivergent individuals with ADHD, autism, dyslexia, dyscalculia, or dyspraxia—may struggle silently without expressing confusion.

Most existing platforms:
- Do not detect real-time learning difficulty
- Continue delivering content at a fixed pace regardless of learner state
- Depend on user interaction (questions, inputs, or commands)
- Lack long-term adaptation across sessions

Additionally, modern AI-based learning tools often rely on cloud systems, raising concerns around privacy, latency, and continuous monitoring—especially when using webcam or voice-based interaction.

This creates a gap between:
> the learner’s real-time cognitive state  
> and the system’s ability to actively teach and adapt

---

## Our Approach

We propose an **offline, privacy-first adaptive learning system** that behaves like a **live AI tutor**.

The system continuously observes learner behavior, understands study material, actively communicates using voice, and dynamically adapts its teaching style in real time—without waiting for explicit user input.

It acts like a tutor that:
> observes → understands → speaks → adapts → guides → repeats

---

## 🔍 Behavioral Awareness (Passive Observation)

The system uses webcam-based signals to understand learner engagement:

- Presence detection
- Gaze direction (on-screen vs off-screen)
- Movement patterns (restlessness, inactivity)

From these signals, it infers **learning states** such as:
- Focused
- Distracted
- Overloaded
- Low engagement

> Note: The system does not diagnose medical or psychological conditions. It adapts only based on observable behavior patterns.

---

## 📚 Content Understanding

The system accepts learning materials such as:
- PDFs
- Notes
- Text documents

It processes and structures content by:
- Extracting clean text
- Splitting into logical sections
- Identifying key concepts

A local AI model (e.g., :contentReference[oaicite:0]{index=0}) is used to:
- Summarize content
- Generate step-by-step explanations
- Reframe content in multiple teaching styles
- Simplify or expand explanations based on learner state

---

## 🎤 Voice-Driven Tutor Interaction

Voice is a **core part of the system**, making the tutor feel alive and interactive.

The system:
- Automatically explains concepts using speech
- Speaks guidance when the learner appears stuck or distracted
- Asks contextual questions such as:
  - “Should I slow this down?”
  - “Do you want a simpler explanation?”
  - “Does this part make sense?”

Voice is not dependent on user input—it is used to actively teach and guide.

Optional speech input allows learners to respond naturally when needed.

---

## ⚡ Real-Time Adaptive Teaching

The system runs a continuous loop:
observe → infer → adapt → respond → speak → repeat


Based on inferred learning state:

- **Focused** → continues detailed explanation  
- **Distracted** → simplifies and shortens content  
- **Overloaded** → breaks content into step-by-step guidance  
- **Low engagement** → rephrases or changes explanation style  

Adaptation happens continuously, making the system feel responsive and alive.

---

## 🧠 Continuous Personalization

The system improves over time by learning from each session:

It tracks:
- Attention patterns
- Time spent per concept
- Repeated sections
- Difficulty areas

It builds a lightweight learner profile:
- Preferred explanation style
- Optimal pacing
- Common struggle points

This allows future sessions to:
- Adapt faster
- Teach more effectively
- Reduce cognitive overload

---

## ♿ Accessibility by Design

The system is designed to support diverse learning needs:

- **ADHD** → short, structured content with pacing control  
- **Autism** → predictable, structured explanations  
- **Dyslexia** → simplified language and readability-focused formatting  
- **Dyscalculia** → step-by-step breakdown of problems  
- **Dyspraxia** → reduced dependency on manual input through voice  

Voice interaction further improves accessibility by reducing reliance on reading or typing.

---

## 🔒 Offline & Privacy-First Design

- Fully runs on-device
- No cloud dependency
- Webcam and voice data processed locally
- No external data sharing

This ensures privacy while enabling continuous observation and adaptation.

---

## 💡 Key Idea

A system that behaves like a **live adaptive tutor**:

> It observes the learner, understands their state, speaks and teaches proactively, and continuously adapts its explanation style—just like a human tutor who never waits to be asked.

---

## ✅ Implementation Status

### Phase 1: Behavioral Observation (COMPLETED)
- ✅ **Webcam Integration** (OpenCV + MediaPipe)
  - Real-time face detection using MediaPipe FaceMesh
  - Pose tracking for body movement detection
  - 30 FPS processing on 640x480 stream
  
- ✅ **Signal Detection**
  - Presence detection (face in frame)
  - Gaze direction tracking (on-screen vs away)
  - Movement level classification (low/medium/high)
  - Attention state inference (focused/distracted)
  
- ✅ **Continuous Processing**
  - Background thread runs observer loop
  - Signals updated every 2 seconds
  - Thread-safe signal access with locks

### Phase 2: Learning State Detection (COMPLETED)
- ✅ **Rule-Based State Machine**
  - **Focused**: Present + on-screen gaze + low movement (confidence: 0.9)
  - **Distracted**: Looking away for 5+ seconds (confidence: 0.6-0.85)
  - **Overloaded**: High movement + off-screen (confidence: 0.65-0.88)
  - **Low Engagement**: Absent or inactive 10+ seconds (confidence: 0.5-0.95)
  
- ✅ **Temporal Tracking**
  - Duration-based confidence scoring
  - State history maintained (50 last states)
  - Trend detection (stable/changing)
  
- ✅ **API Endpoints**
  - `GET /learning-state` - Current state + confidence
  - `GET /learning-state/summary` - Detailed analysis
  - `/signals` and `/signals/stream` include learning state
  - `GET /api/learning-states/info` - Documentation

### Phase 3: Dashboard (COMPLETED)
- ✅ **Real-Time Display**
  - Live learning state with color coding
  - Confidence percentage
  - Behavioral signals display
  - State change log
  
- ✅ **SSE Streaming**
  - `/signals/stream` endpoint
  - Auto-reconnecting client
  - Updates every 1 second

### Upcoming Phases

#### Phase 3: Content Understanding
- Load and parse learning materials (PDFs, notes, text)
- Split into logical sections
- Extract key concepts
- Generate multi-style explanations

#### Phase 4: Voice-Driven Interaction
- Text-to-speech for automatic explanations
- Speech recognition for learner responses
- Context-aware follow-up questions
- Proactive guidance based on learning state

#### Phase 5: Adaptive Teaching Loop
- State-based content selection
- Pacing adjustment
- Explanation style switching
- Real-time intervention

#### Phase 6: Personalization
- Session-based learner profiles
- Preference learning
- Difficulty calibration
- Long-term improvement tracking

---

## 🚀 Getting Started

### Requirements
```
fastapi==0.109.0
uvicorn==0.27.0
opencv-python==4.10.0.84
mediapipe==0.10.14
numpy==2.1.1
```

### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Run the System
```bash
cd backend
uvicorn main:app --reload
```

### Access Dashboard
Open browser to: `http://127.0.0.1:8000`

### Monitor API
```bash
# Real-time learning state
curl http://127.0.0.1:8000/learning-state

# Detailed analysis
curl http://127.0.0.1:8000/learning-state/summary

# All signals
curl http://127.0.0.1:8000/signals
```

---

## 📊 System Architecture

```
Webcam Input
    ↓
[BehavioralObserver]
├─ Face Detection (MediaPipe FaceMesh)
├─ Gaze Analysis (Eye landmarks)
├─ Pose Tracking (Shoulder/hip movement)
    ↓
Behavioral Signals
├─ presence (bool)
├─ gaze_on_screen (bool)
├─ movement_level (low/medium/high)
└─ attention_state (focused/distracted)
    ↓
[LearningStateDetector]
├─ Temporal Tracking
├─ Rule-Based Logic
└─ Confidence Scoring
    ↓
Learning State
├─ State (focused/distracted/overloaded/low_engagement)
├─ Confidence (0.0-1.0)
└─ Trend
    ↓
[Dashboard + API]
├─ Real-time visualization
├─ RESTful endpoints
└─ SSE streaming
```

---

## 📝 Documentation Files

- `FoE.md` - This file (project vision and implementation roadmap)
- `LEARNING_STATE_SYSTEM.md` - Detailed learning state detection logic and rules
- `backend/utils/learning_state.py` - Learning state detector implementation
- `backend/utils/observer.py` - Behavioral signal detection
- `backend/main.py` - FastAPI application and dashboard
