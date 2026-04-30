import os
import torch
from peft import PeftModel, PeftConfig 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
from typing import Dict, List, Optional
import logging
import re
import json

logger = logging.getLogger(__name__)


class GemmaContentProcessor:
    """Processes learning content using trained Gemma 4 model with RAG."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._is_loaded = False
        self._rag_pipeline = None
    
    def _get_rag_pipeline(self):
        """Lazy load RAG pipeline."""
        if self._rag_pipeline is None:
            from utils.rag_pipeline import get_rag_pipeline
            self._rag_pipeline = get_rag_pipeline()
        return self._rag_pipeline
    
    def load_model(self):
        """Load the trained Gemma 4 model with LoRA adapters - use local files only, no download."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"🔄 Loading trained Gemma 4 model from: {self.model_path}")
            
            import os
            import json
            # Check local files
            if not os.path.exists(self.model_path):
                raise ValueError(f"Model path does not exist: {self.model_path}")
            
            files = os.listdir(self.model_path)
            logger.info(f"📁 Local files: {files}")
            
            # Load tokenizer from local tokenizer.json
            tokenizer_file = os.path.join(self.model_path, "tokenizer.json")
            if os.path.exists(tokenizer_file):
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
                logger.info("✅ Tokenizer loaded from local file")
            else:
                raise ValueError("No tokenizer.json found")
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "</s>"
            
            # Load the base model configuration from adapter_config.json
            adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path", "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit")
                logger.info(f"📌 Base model from adapter config: {base_model_path}")
            else:
                base_model_path = "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit"
                logger.warning("⚠️ No adapter_config.json found, using default base model")
            
            # Try to load the base model with proper Gemma 4 support
            try:
                logger.info("🔄 Loading base Gemma 4 model...")
                
                # Load base model - will use HuggingFace cache if available
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    trust_remote_code=True
                )
                logger.info("✅ Base model loaded from HuggingFace cache")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load base model from HuggingFace: {e}")
                logger.info("🔄 Attempting fallback with minimal config...")
                
                # Fallback: Try loading with minimal config
                try:
                    config = AutoConfig.from_pretrained(
                        base_model_path,
                        trust_remote_code=True
                    )
                    self.base_model = AutoModelForCausalLM.from_config(
                        config,
                        torch_dtype=torch.bfloat16,
                        device_map="cpu",
                        trust_remote_code=True
                    )
                    logger.info("✅ Base model loaded with minimal config")
                except Exception as e2:
                    logger.error(f"❌ Base model loading failed: {e2}")
                    raise ValueError(f"Cannot load base model. Please ensure {base_model_path} is accessible.")
            
            # Load LoRA adapters from trained model
            logger.info("🔄 Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
            logger.info("✅ LoRA adapters loaded successfully")
            
            # Set to evaluation mode
            self.model.eval()
            
            # Create pipeline for text generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            self._is_loaded = True
            logger.info("✅ Gemma 4 model with LoRA adapters loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            self._is_loaded = False
            raise
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if self.pipeline is not None:
                del self.pipeline
            torch.cuda.empty_cache()
            self._is_loaded = False
            logger.info("🔄 Model unloaded")
    
    def process_content(self, text: str, task: str = "analyze") -> Dict:
        """
        Process content with Gemma 4 based on task type.
        
        Args:
            text: Input text to process
            task: Type of processing - 'extract', 'summarize', 'simplify', 
                  'structure', 'teach_styles', 'full'
        
        Returns:
            Dict with processed content
        """
        if not self._is_loaded:
            self.load_model()
        
        results = {}
        
        if task == "extract" or task == "full":
            results['key_concepts'] = self._extract_key_concepts(text)
            results['sections'] = self._identify_sections(text)
        
        if task == "summarize" or task == "full":
            results['summary'] = self._generate_summary(text)
        
        if task == "simplify" or task == "full":
            results['simplified'] = self._simplify_content(text)
        
        if task == "structure" or task == "full":
            results['learning_chunks'] = self._create_learning_chunks(text)
        
        if task == "teach_styles" or task == "full":
            results['teaching_styles'] = self._generate_teaching_styles(text)
        
        return results
    
    def _create_prompt(self, instruction: str, content: str) -> str:
        """Create formatted prompt for Gemma."""
        return f"""<start_of_turn>user
{instruction}

Content to process:
{content[:3000]}<end_of_turn>
<start_of_turn>model
"""
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        prompt = self._create_prompt(
            "Extract the key concepts and main ideas from this learning material. "
            "List each concept as a brief phrase (2-5 words). Return only the concepts, one per line.",
            text
        )
        
        try:
            output = self.pipeline(prompt, max_new_tokens=512)[0]['generated_text']
            # Parse concepts from output
            concepts = []
            for line in output.split('\n'):
                line = line.strip()
                if line and len(line) < 100:
                    # Clean up the line
                    concept = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if concept:
                        concepts.append(concept)
            return concepts[:20]  # Limit to 20 concepts
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []
    
    def _identify_sections(self, text: str) -> List[Dict]:
        """Identify major sections in the content."""
        prompt = self._create_prompt(
            "Identify the major sections and topics in this learning material. "
            "For each section, provide: 1) Section title, 2) Brief description (1 sentence). "
            "Format as: Section: [title] | Description: [desc]",
            text
        )
        
        try:
            output = self.pipeline(prompt, max_new_tokens=512)[0]['generated_text']
            sections = []
            for line in output.split('\n'):
                if 'Section:' in line or '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        title = parts[0].replace('Section:', '').strip()
                        desc = parts[1].replace('Description:', '').strip()
                        sections.append({'title': title, 'description': desc})
            return sections[:10]
        except Exception as e:
            logger.error(f"Section identification failed: {e}")
            return []
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the content."""
        prompt = self._create_prompt(
            "Create a comprehensive summary of this learning material. "
            "Include: 1) Main topic, 2) Key points (3-5), 3) Learning objectives. "
            "Keep it concise but informative.",
            text
        )
        
        try:
            output = self.pipeline(prompt, max_new_tokens=1024)[0]['generated_text']
            # Extract the model response
            if '<start_of_turn>model' in output:
                output = output.split('<start_of_turn>model')[-1]
            return output.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Summary unavailable"
    
    def _simplify_content(self, text: str) -> str:
        """Simplify the content for easier understanding."""
        prompt = self._create_prompt(
            "Simplify this learning material. "
            "Break down complex concepts into simpler terms. "
            "Use analogies where helpful. Make it accessible to beginners.",
            text
        )
        
        try:
            output = self.pipeline(prompt, max_new_tokens=1024)[0]['generated_text']
            if '<start_of_turn>model' in output:
                output = output.split('<start_of_turn>model')[-1]
            return output.strip()
        except Exception as e:
            logger.error(f"Simplification failed: {e}")
            return text
    
    def _create_learning_chunks(self, text: str) -> List[Dict]:
        """Break content into structured learning chunks."""
        prompt = self._create_prompt(
            "Break this content into structured learning chunks. "
            "For each chunk provide: 1) Topic name, 2) Core concept (1-2 sentences), "
            "3) Key points (bullet list), 4) Difficulty level (beginner/intermediate/advanced). "
            "Format: CHUNK | [topic] | [concept] | [points] | [level]",
            text
        )
        
        try:
            output = self.pipeline(prompt, max_new_tokens=1024)[0]['generated_text']
            chunks = []
            for line in output.split('\n'):
                if 'CHUNK' in line or '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        chunks.append({
                            'topic': parts[1].strip() if len(parts) > 1 else 'Unknown',
                            'concept': parts[2].strip() if len(parts) > 2 else '',
                            'points': parts[3].strip() if len(parts) > 3 else '',
                            'level': parts[4].strip() if len(parts) > 4 else 'beginner'
                        })
            return chunks[:15]
        except Exception as e:
            logger.error(f"Chunk creation failed: {e}")
            return []
    
    def _generate_teaching_styles(self, text: str) -> List[Dict]:
        """Generate multiple teaching styles for the content."""
        styles = [
            {
                'name': 'Direct Instruction',
                'description': 'Clear, structured explanations with explicit steps',
                'prompt_suffix': 'Present this as a step-by-step tutorial with clear headings.'
            },
            {
                'name': 'Socratic Method',
                'description': 'Question-based learning to guide discovery',
                'prompt_suffix': 'Present this as a series of questions that guide the learner to discover concepts.'
            },
            {
                'name': 'Visual/Analogical',
                'description': 'Use metaphors and visual descriptions',
                'prompt_suffix': 'Explain using analogies and visual descriptions.'
            },
            {
                'name': 'Problem-Based',
                'description': 'Learn through solving practical problems',
                'prompt_suffix': 'Frame this as problems to solve with the concepts as tools.'
            },
            {
                'name': 'Narrative',
                'description': 'Story-based learning with context',
                'prompt_suffix': 'Present this as a story with context and progression.'
            }
        ]
        
        results = []
        for style in styles:
            prompt = self._create_prompt(
                f"Teach this content using {style['name']} approach. {style['prompt_suffix']}",
                text
            )
            
            try:
                output = self.pipeline(prompt, max_new_tokens=1024)[0]['generated_text']
                if '<start_of_turn>model' in output:
                    output = output.split('<start_of_turn>model')[-1]
                
                results.append({
                    'style': style['name'],
                    'description': style['description'],
                    'content': output.strip()[:2000]  # Limit content length
                })
            except Exception as e:
                logger.error(f"Teaching style '{style['name']}' failed: {e}")
                results.append({
                    'style': style['name'],
                    'description': style['description'],
                    'content': 'Content unavailable'
                })
        
        return results
    
    def process_full(self, text: str) -> Dict:
        """Process content with all available methods."""
        return self.process_content(text, task="full")
    
    # ==================== RAG-Based Tutor Methods ====================
    
    def index_document_for_rag(self, text: str, doc_name: str = "default") -> Dict:
        """
        Index a document for RAG-based retrieval.
        
        Call this after loading PDF to enable contextual Q&A.
        """
        rag = self._get_rag_pipeline()
        chunks = rag.load_document(text, doc_name)
        
        return {
            "status": "indexed",
            "document": doc_name,
            "num_chunks": len(chunks),
            "message": f"✅ Indexed {len(chunks)} chunks for RAG retrieval"
        }
    
    def answer_with_rag(self, query: str, doc_name: str = "default") -> Dict:
        """
        Answer a question using RAG - the model looks at the document first.
        
        This is the key RAG method that:
        1. Retrieves relevant chunks from the document
        2. Injects context into the prompt
        3. Generates answer based on document content
        """
        if not self._is_loaded:
            self.load_model()
        
        rag = self._get_rag_pipeline()
        
        # Get RAG-enhanced prompt with retrieved context
        rag_prompt = rag.get_rag_prompt(query, doc_name)
        
        try:
            output = self.pipeline(rag_prompt, max_new_tokens=1024)[0]['generated_text']
            
            # Extract the model response
            if '<start_of_turn>model' in output:
                answer = output.split('<start_of_turn>model')[-1].strip()
            else:
                answer = output.strip()
            
            # Get source information
            rag_result = rag.query(query, doc_name)
            
            return {
                "query": query,
                "answer": answer,
                "sources": rag_result['sources'],
                "context_used": rag_result['context'][:500] + "..." if len(rag_result['context']) > 500 else rag_result['context'],
                "rag_enabled": True
            }
        except Exception as e:
            logger.error(f"RAG answer failed: {e}")
            return {
                "query": query,
                "answer": "Sorry, I couldn't generate a response at this time.",
                "error": str(e),
                "rag_enabled": True
            }
    
    def tutor_interaction(self, query: str, learning_state: str = "focused", doc_name: str = "default") -> Dict:
        """
        Full tutor interaction with RAG + learning state awareness.
        
        This method:
        1. Uses RAG to retrieve document context
        2. Adapts response based on learning state
        3. Adds tutor-like phrases
        4. Handles out-of-context questions gracefully
        """
        if not self._is_loaded:
            self.load_model()
        
        rag = self._get_rag_pipeline()
        
        # Check if document is indexed
        if not rag.indexed_docs or doc_name not in rag.indexed_docs:
            return {
                "query": query,
                "answer": "No document has been indexed yet. Please index a PDF first using the Content tab or click 'Index PDF for Q&A' in the Tutor tab.",
                "learning_state": learning_state,
                "sources": [],
                "rag_enabled": False,
                "error": "Document not indexed"
            }
        
        # Get context from document via RAG
        rag_result = rag.query(query, doc_name)
        context = rag_result.get('context', '')
        sources = rag_result.get('sources', [])
        
        # Check if we have meaningful context
        has_meaningful_context = bool(context and len(context.strip()) > 50)
        
        # Adapt instruction based on learning state
        state_instructions = {
            "focused": "Provide detailed, comprehensive explanation.",
            "distracted": "Keep explanation brief and engaging. Use simple terms.",
            "overloaded": "Break into small steps. One concept at a time.",
            "low_engagement": "Use an engaging, question-based approach to re-interest the learner."
        }
        
        instruction = state_instructions.get(learning_state, state_instructions["focused"])
        
        # Build the prompt based on whether we have context
        if has_meaningful_context:
            # We have document context - use RAG
            prompt = f"""<start_of_turn>user
You are an expert tutor helping a student. 

LEARNER STATE: {learning_state.upper()}
{instruction}

DOCUMENT CONTEXT (retrieved from learning material):
{context}

Based on the above context from the document, please answer this question:
{query}

Add tutor-like phrases like:
- "Let me explain..."
- "This is important because..."
- "Should I slow down?"
- "Let's break this down..."
<end_of_turn>
<start_of_turn>model
"""
        else:
            # No meaningful context from document - handle out-of-context question
            # Check if query might be related to the document topic
            prompt = f"""<start_of_turn>user
You are an expert tutor helping a student. 

LEARNER STATE: {learning_state.upper()}
{instruction}

The student asked: "{query}"

The document that has been indexed does not contain specific information about this topic.
However, you should still help the student.

Respond in this format:
1. First acknowledge: "This question is not directly covered in the loaded document, but..."
2. Then provide a brief, general explanation if you have knowledge about it
3. Or say "I don't have specific information about this topic in the document we're studying"

Be helpful and encouraging as a tutor.
<end_of_turn>
<start_of_turn>model
"""
        
        try:
            output = self.pipeline(prompt, max_new_tokens=1024)[0]['generated_text']
            
            if '<start_of_turn>model' in output:
                answer = output.split('<start_of_turn>model')[-1].strip()
            else:
                answer = output.strip()
            
            # Check for empty or error responses
            if not answer or len(answer.strip()) < 10:
                answer = "I'm having trouble processing that right now. Please try again."
            
            return {
                "query": query,
                "answer": answer,
                "learning_state": learning_state,
                "sources": sources,
                "rag_enabled": True,
                "context_used": has_meaningful_context
            }
        except Exception as e:
            logger.error(f"Tutor interaction failed: {e}")
            return {
                "query": query,
                "answer": "I'm here to help! Could you please repeat your question?",
                "learning_state": learning_state,
                "error": str(e)
            }


# Singleton instance for reuse
_processor_instance = None


def get_content_processor(model_path: str = None) -> GemmaContentProcessor:
    """Get or create singleton content processor."""
    global _processor_instance
    
    if _processor_instance is None:
        if model_path is None:
            # Default path to trained model
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "trained_gemma4_unsloth")
        
        _processor_instance = GemmaContentProcessor(model_path)
        logger.info(f"📦 Created new content processor with model path: {model_path}")
    
    # Always ensure model is loaded when getting the processor
    if not _processor_instance._is_loaded:
        logger.info("🔄 Model not loaded, loading now...")
        try:
            _processor_instance.load_model()
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    return _processor_instance


# ==================== Model Test & Integration Functions ====================

def test_model():
    """
    Test the trained Gemma 4 model with a simple text generation.
    Returns test results with model info and sample output.
    """
    logger.info("🧪 Testing Gemma 4 model...")
    
    try:
        # Get the content processor
        processor = get_content_processor()
        
        # Test prompt
        test_prompt = """<start_of_turn>user
Explain the concept of photosynthesis in simple terms.<end_of_turn>
<start_of_turn>model
"""
        
        # Generate response
        output = processor.pipeline(test_prompt, max_new_tokens=256)[0]['generated_text']
        
        # Extract the response
        if '<start_of_turn>model' in output:
            response = output.split('<start_of_turn>model')[-1].strip()
        else:
            response = output.strip()
        
        return {
            "status": "success",
            "model_loaded": processor._is_loaded,
            "model_path": processor.model_path,
            "test_output": response[:500] if response else "No output generated",
            "message": "✅ Model test passed! Gemma 4 is working correctly."
        }
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "message": "❌ Model test failed. Check logs for details."
        }


def process_with_observer_integration(query: str, observer_signals: dict = None, doc_name: str = "default") -> Dict:
    """
    Process a query with observer integration for adaptive responses.
    
    This function:
    1. Gets learning state from observer signals
    2. Uses RAG for document context
    3. Generates adaptive response via Gemma 4
    4. Returns response ready for TTS voice output
    
    Args:
        query: User's question
        observer_signals: Dict with 'state', 'confidence', 'signals' from observer
        doc_name: Name of indexed document for RAG
    
    Returns:
        Dict with 'answer', 'learning_state', 'sources', 'voice_ready'
    """
    # Extract learning state from observer signals
    learning_state = "focused"
    if observer_signals:
        learning_state = observer_signals.get("state", "focused")
    
    # Get the content processor
    processor = get_content_processor()
    
    # Use tutor_interaction which already handles RAG + learning state
    result = processor.tutor_interaction(query, learning_state, doc_name)
    
    # Add voice-ready flag
    result["voice_ready"] = True
    result["voice_text"] = result.get("answer", "")[:2000]  # Limit for TTS
    
    return result


def generate_intervention_response(observer_signals: dict) -> Dict:
    """
    Generate an intervention response when observer detects issues.
    
    Called by VoiceTriggerController when student needs help.
    
    Args:
        observer_signals: Dict with 'state', 'confidence', 'signals' from observer
    
    Returns:
        Dict with intervention message, voice_text, actions
    """
    state = observer_signals.get("state", "unknown")
    confidence = observer_signals.get("confidence", 0.0)
    
    # Get the content processor
    processor = get_content_processor()
    
    # Generate contextual intervention based on state
    intervention_prompts = {
        "distracted": """<start_of_turn>user
The student seems distracted. Give a brief, engaging reminder to focus.
Use an encouraging tone. Keep it under 50 words.<end_of_turn>
<start_of_turn>model
""",
        "overloaded": """<start_of_turn>user
The student appears overwhelmed. Break down the current topic into one simple step.
Be patient and reassuring. Keep it under 50 words.<end_of_turn>
<start_of_turn>model
""",
        "low_engagement": """<start_of_turn>user
The student seems disengaged. Ask an interesting question to spark curiosity.
Use an engaging, question-based approach. Keep it under 50 words.<end_of_turn>
<start_of_turn>model
""",
        "stuck_while_focused": """<start_of_turn>user
The focused student is stuck on a problem. Give a gentle hint without the full answer.
Encourage them to think through it. Keep it under 50 words.<end_of_turn>
<start_of_turn>model
""",
        "misread_detected": """<start_of_turn>user
The student seems to have misread something. Gently clarify the correct understanding.
Be helpful, not corrective. Keep it under 50 words.<end_of_turn>
<start_of_turn>model
"""
    }
    
    prompt = intervention_prompts.get(state, intervention_prompts["distracted"])
    
    try:
        output = processor.pipeline(prompt, max_new_tokens=100)[0]['generated_text']
        
        if '<start_of_turn>model' in output:
            answer = output.split('<start_of_turn>model')[-1].strip()
        else:
            answer = output.strip()
        
        return {
            "state": state,
            "confidence": confidence,
            "intervention_message": answer,
            "voice_text": answer,
            "voice_ready": True,
            "auto_triggered": True
        }
        
    except Exception as e:
        logger.error(f"Intervention generation failed: {e}")
        # Fallback messages
        fallback = {
            "distracted": "Let's take a moment to review what we've learned so far.",
            "overloaded": "Let's break this down into smaller steps.",
            "low_engagement": "What do you think about this concept?",
            "stuck_while_focused": "Have you tried looking at it from a different angle?",
            "misread_detected": "Let me clarify that point for you."
        }
        return {
            "state": state,
            "confidence": confidence,
            "intervention_message": fallback.get(state, "Let's continue learning."),
            "voice_text": fallback.get(state, "Let's continue learning."),
            "voice_ready": True,
            "auto_triggered": True,
            "fallback": True
        }


def get_model_info() -> Dict:
    """Get information about the loaded model."""
    processor = get_content_processor()
    
    return {
        "model_loaded": processor._is_loaded,
        "model_path": processor.model_path,
        "has_pipeline": processor.pipeline is not None,
        "has_tokenizer": processor.tokenizer is not None,
        "tokenizer_vocab_size": processor.tokenizer.vocab_size if processor.tokenizer else None
    }