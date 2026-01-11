"""
High-parameter submitter agent for compiler.
Handles rigor enhancement mode.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.models import CompilerSubmission
from backend.shared.config import system_config, rag_config
from backend.shared.json_parser import parse_json
from backend.aggregator.validation.json_validator import json_validator
from backend.compiler.prompts.rigor_prompts import build_rigor_prompt
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import paper_memory
from backend.compiler.core.compiler_rag_manager import compiler_rag_manager

logger = logging.getLogger(__name__)


def _normalize_string_field(value) -> str:
    """
    Normalize string field from LLM response.
    Some LLMs incorrectly return strings as lists.
    
    Args:
        value: Raw value from JSON (could be str, list, or other)
    
    Returns:
        Normalized string value
    """
    if isinstance(value, list):
        # LLM returned list - join into single string
        logger.warning(f"LLM returned field as list (length {len(value)}), converting to string")
        return " ".join(str(item) for item in value if item)
    elif isinstance(value, str):
        return value
    elif value is None:
        return ""
    else:
        # Fallback: convert to string
        logger.warning(f"LLM returned field as {type(value)}, converting to string")
        return str(value)


class HighParamSubmitter:
    """
    High-parameter, low-context submitter for compiler.
    
    Mode:
    - rigor: Enhance scientific rigor of the paper
    """
    
    def __init__(self, model_name: str, user_prompt: str, websocket_broadcaster: Optional[Callable] = None):
        self.model_name = model_name
        self.user_prompt = user_prompt
        self.websocket_broadcaster = websocket_broadcaster
        self._initialized = False
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "compiler_high_param"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"comp_hp_{self.task_sequence:03d}"
    
    async def initialize(self) -> None:
        """Initialize submitter."""
        if self._initialized:
            return
        
        # Set context window from system config
        self.context_window = system_config.compiler_high_param_context_window
        self.max_output_tokens = system_config.compiler_high_param_max_output_tokens
        self.available_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        
        self._initialized = True
        logger.info(f"High-param submitter initialized with model: {self.model_name}")
        logger.info(f"Context budget: {self.available_input_tokens} tokens (window: {self.context_window})")
    
    async def _generate_with_truncation_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        operation_name: str,
        task_id: str,
        max_tokens_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate completion with INFINITE retry if truncated.
        Will NEVER give up - keeps retrying until success.
        
        Args:
            messages: Chat messages
            temperature: Temperature for generation
            operation_name: Name of operation (for logging)
            task_id: Task ID for API client manager tracking
            max_tokens_override: Optional max tokens override
        
        Returns:
            Response dict (only when successful)
        """
        retry_count = 0
        current_max_tokens = max_tokens_override
        
        while True:  # INFINITE RETRY - NEVER GIVE UP
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=current_max_tokens
            )
            
            # Check if response was truncated
            finish_reason = response.get("choices", [{}])[0].get("finish_reason", "")
            
            if finish_reason == "length":
                retry_count += 1
                usage = response.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                
                logger.warning(
                    f"{operation_name}: Response truncated at {completion_tokens} tokens "
                    f"(finish_reason='length'). Retry #{retry_count} (will NEVER give up)"
                )
                
                # Try to parse anyway - might still be valid JSON
                # Extract content from either 'content' or 'reasoning' field
                message = response["choices"][0]["message"]
                llm_output = message.get("content", "") or message.get("reasoning", "")
                if llm_output.strip().endswith("}"):
                    logger.info(f"{operation_name}: Truncated response appears complete, attempting parse")
                    return response
                
                # If we have max_tokens set and hit the limit, increase it by 50%
                if current_max_tokens:
                    current_max_tokens = int(current_max_tokens * 1.5)
                    logger.info(f"{operation_name}: Increasing max_tokens to {current_max_tokens} and retrying")
                else:
                    # If no limit set, something else is wrong - return anyway and let parser handle
                    logger.warning(f"{operation_name}: No max_tokens set but still truncated, returning response for parse attempt")
                    return response
                
                # Log every 10 retries to show we're still trying
                if retry_count % 10 == 0:
                    logger.warning(
                        f"{operation_name}: Still retrying after {retry_count} attempts. "
                        f"Current max_tokens: {current_max_tokens}. WILL NEVER STOP."
                    )
                
                continue  # Keep trying forever
            
            # Success - not truncated
            if retry_count > 0:
                logger.info(f"{operation_name}: Success after {retry_count} truncation retries")
            return response
    
    async def submit_rigor_enhancement(self) -> Optional[CompilerSubmission]:
        """
        Submit rigor enhancement (or no-op if not needed).
        
        Returns:
            CompilerSubmission if enhancement needed, None otherwise
        """
        logger.info("Starting rigor enhancement submission generation...")
        
        try:
            # Get current outline and paper
            logger.info("Loading outline and paper state...")
            current_outline = await outline_memory.get_outline()
            current_paper = await paper_memory.get_paper()
            logger.info(f"State loaded: outline={len(current_outline)} chars, paper={len(current_paper)} chars")
            
            # For rigor mode with low-context model:
            # - Outline: ALWAYS fully injected (crucial for paper structure understanding)
            # - Paper: RAG retrieval for relevant sections (context-efficient)
            
            from backend.shared.utils import count_tokens
            max_allowed_tokens = rag_config.get_available_input_tokens(system_config.compiler_high_param_context_window, system_config.compiler_high_param_max_output_tokens)
            
            # Try initial RAG retrieval - may overflow if outline + system prompts are large
            try:
                # Retrieve relevant paper sections via RAG (uses paper as query context)
                logger.info("Retrieving relevant paper sections via RAG for rigor analysis...")
                context_pack = await compiler_rag_manager.retrieve_for_mode(
                    query=self.user_prompt + " " + current_paper[-1000:],  # Use end of paper to guide retrieval
                    mode="rigor"
                )
                logger.info(f"RAG retrieval complete: {len(context_pack.text)} chars retrieved")
                
                # Build prompt with FULL outline injection + RAG'd paper content
                logger.info("Building rigor enhancement prompt (full outline + RAG'd paper)...")
                prompt = await build_rigor_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,  # FULL INJECTION - critical for structure
                    current_paper=context_pack.text  # RAG'd relevant sections
                )
                logger.info(f"Prompt built: {len(prompt)} chars")
                
                # CRITICAL: Verify actual prompt size fits in context window
                actual_prompt_tokens = count_tokens(prompt)
                
                if actual_prompt_tokens > max_allowed_tokens:
                    raise ValueError(f"Prompt too large: {actual_prompt_tokens} tokens > {max_allowed_tokens} max")
                
                logger.debug(f"Rigor mode prompt: {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
                
            except ValueError as e:
                if "Prompt too large" not in str(e):
                    raise  # Different error, propagate
                
                # Context overflow detected - calculate reduced RAG budget
                logger.warning(f"Rigor mode: Initial prompt too large. Calculating reduced RAG budget...")
                
                # Calculate mandatory components (outline + system prompts + user prompt)
                mandatory_tokens = count_tokens(
                    await build_rigor_prompt(self.user_prompt, current_outline, "")  # Empty paper
                )
                
                # Calculate reduced RAG budget with safety buffer
                remaining_budget = max_allowed_tokens - mandatory_tokens - 200  # 200 token safety buffer
                
                if remaining_budget < 500:  # Minimum viable RAG content
                    raise ValueError(
                        f"Context window too small for rigor mode: outline + system prompts require "
                        f"{mandatory_tokens} tokens, only {max_allowed_tokens} available. "
                        f"Increase compiler_high_param_context_window or reduce outline size."
                    )
                
                # Retry with reduced budget
                logger.warning(
                    f"Rigor mode: Retrying RAG retrieval with reduced budget: {remaining_budget} tokens "
                    f"(mandatory: {mandatory_tokens}, total: {max_allowed_tokens})"
                )
                context_pack = await compiler_rag_manager.retrieve_for_mode(
                    query=self.user_prompt + " " + current_paper[-1000:],
                    mode="rigor",
                    max_tokens=remaining_budget
                )
                
                # Rebuild prompt with reduced content
                prompt = await build_rigor_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=context_pack.text
                )
                
                # Re-validate
                actual_prompt_tokens = count_tokens(prompt)
                logger.info(
                    f"Rigor mode: Adjusted prompt to {actual_prompt_tokens} tokens "
                    f"(budget: {max_allowed_tokens}, paper content reduced)"
                )
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Get completion with truncation retry via api_client_manager (handles boost and fallback)
            logger.info(f"Generating LLM completion via api_client_manager (task_id={task_id})...")
            response = await self._generate_with_truncation_retry(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic generation - evolving context provides diversity
                operation_name="rigor",
                task_id=task_id,
                max_tokens_override=system_config.compiler_high_param_max_output_tokens  # User-configurable, will auto-increase if truncated
            )
            # Extract content from either 'content' or 'reasoning' field
            # Some reasoning models (e.g., DeepSeek R1, certain GPT variants) output JSON in 'reasoning' field
            message = response["choices"][0]["message"]
            llm_output = message.get("content", "") or message.get("reasoning", "")
            logger.info(f"LLM completion received: {len(llm_output)} chars")
            
            # Parse response with retry
            logger.info("Parsing JSON response...")
            data = await self._parse_json_response_with_retry(llm_output, prompt, task_id)
            logger.info("JSON parsed successfully")
            
            if not data:
                raise ValueError("Failed to parse JSON response from rigor enhancement")
            
            # Handle case where model returns array instead of single object
            if isinstance(data, list):
                if len(data) == 0:
                    logger.warning("Rigor enhancement returned empty array, treating as no enhancement needed")
                    # Notify task completed (even when no enhancement needed)
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    return None
                logger.warning(f"Rigor enhancement returned array of {len(data)} objects, using first object only")
                data = data[0]
            
            # Check if enhancement needed
            needs_enhancement = data.get("needs_enhancement", False)
            
            if not needs_enhancement:
                # Notify task completed (even when no enhancement needed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                logger.info("Rigor enhancement not needed")
                return None
            
            # Create submission
            # Use new_string as content for logging
            new_string_content = _normalize_string_field(data.get("new_string", ""))
            
            submission = CompilerSubmission(
                submission_id=str(uuid.uuid4()),
                mode="rigor",
                content=new_string_content,  # Use new_string as the content
                operation=data.get("operation", "replace"),
                old_string=_normalize_string_field(data.get("old_string", "")),
                new_string=new_string_content,  # Already normalized above
                reasoning=data.get("reasoning", ""),
                metadata={}
            )
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"Rigor enhancement submission generated: {submission.submission_id}")
            return submission
            
        except Exception as e:
            logger.error(f"Failed to generate rigor enhancement submission: {e}", exc_info=True)
            # Notify task completed (failed but still completed)
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            raise
    
    async def _parse_json_response_with_retry(
        self,
        response: str,
        original_prompt: str,
        task_id: str
    ) -> Optional[dict]:
        """
        Parse JSON response with conversational retry on failure.
        
        Args:
            response: LLM response
            original_prompt: Original prompt sent to LLM (for retry context)
            task_id: Task ID for tracking retry attempt
        
        Returns:
            Parsed JSON dict or None if validation fails after retries
        """
        # Cache model config on first use (only relevant for LM Studio)
        try:
            await lm_studio_client.cache_model_load_config(self.model_name, {
                "context_length": self.context_window,
                "model_path": self.model_name
            })
        except Exception:
            # Silently ignore - only applies to LM Studio models
            pass
        
        # Parse JSON
        try:
            parsed = parse_json(response)
            return parsed
            
        except Exception as parse_error:
            # Not corrupted, just invalid JSON - continue with conversational retry
            valid = False
            parsed = None
            error = str(parse_error)
            
            # Initial parse failed - attempt conversational retry
            logger.info("Compiler high-param submitter (rigor): Initial JSON parse failed, attempting retry")
            logger.debug(f"Parse error: {error}")
        
        # Build retry prompt
        retry_prompt = (
            f"Your previous response could not be parsed as valid JSON.\n\n"
            f"PARSE ERROR: {error}\n\n"
            "JSON ESCAPING RULES FOR LaTeX:\n"
            "LaTeX notation IS ALLOWED - but you must escape it properly in JSON:\n"
            "1. Every backslash in your content needs ONE escape in JSON\n"
            "   - To write \\mathbb{Z} in content, write: \"\\\\mathbb{Z}\" in JSON\n"
            "   - To write \\( and \\), write: \"\\\\(\" and \"\\\\)\" in JSON\n"
            "2. Do NOT double-escape: \\\\\\\\mathbb is WRONG, \\\\mathbb is CORRECT\n"
            "3. For old_string: copy text EXACTLY from the document, just escape backslashes\n"
            "4. Escape quotes inside strings: use \\\" for literal quotes\n"
            "5. Avoid malformed unicode escapes (must be exactly \\uXXXX with 4 hex digits)\n\n"
            "Please provide your response again in valid JSON format:\n"
            "{\n"
            '  "needs_enhancement": true or false,\n'
            '  "operation": "replace | insert_after",\n'
            '  "old_string": "exact text from paper (escape backslashes)",\n'
            '  "new_string": "enhanced text (LaTeX allowed, escape backslashes)",\n'
            '  "content": "full content for logging",\n'
            '  "reasoning": "explanation"\n'
            "}\n\n"
            "Respond with ONLY the JSON object, no markdown, no explanation."
        )
        
        try:
            # CRITICAL FIX: Truncate failed output to prevent context overflow during retry
            from backend.shared.utils import count_tokens
            
            max_failed_output_chars = 2000  # ~500 tokens - enough for error context
            if len(response) > max_failed_output_chars:
                failed_output_preview = response[:max_failed_output_chars] + "\n[...output truncated for retry...]"
            else:
                failed_output_preview = response
            
            # Calculate if conversation fits in context window
            prompt_tokens = count_tokens(original_prompt)
            preview_tokens = count_tokens(failed_output_preview)
            retry_prompt_tokens = count_tokens(retry_prompt)
            conversation_tokens = prompt_tokens + preview_tokens + retry_prompt_tokens
            
            if conversation_tokens > self.available_input_tokens:
                # Too large - just retry with original prompt
                logger.warning(
                    f"Compiler high-param submitter (rigor): Retry conversation too large "
                    f"({conversation_tokens} > {self.available_input_tokens}), using simple retry"
                )
                retry_response = await api_client_manager.generate_completion(
                    task_id=f"{task_id}_retry",
                    role_id=self.role_id,
                    model=self.model_name,
                    messages=[{"role": "user", "content": original_prompt}],
                    temperature=0.0,
                    max_tokens=self.max_output_tokens
                )
            else:
                # Build conversation with truncated failed output
                retry_response = await api_client_manager.generate_completion(
                    task_id=f"{task_id}_retry",  # Track retry attempt
                    role_id=self.role_id,
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": original_prompt},
                        {"role": "assistant", "content": failed_output_preview},
                        {"role": "user", "content": retry_prompt}
                    ],
                    temperature=0.0,  # Deterministic JSON formatting
                    max_tokens=self.max_output_tokens  # Respect max_tokens on retry
                )
            
            if retry_response.get("choices"):
                retry_output = retry_response["choices"][0]["message"]["content"]
                
                try:
                    parsed = parse_json(retry_output)
                    logger.info("Compiler high-param submitter (rigor): Retry succeeded!")
                    return parsed
                except Exception as parse_error:
                    error = str(parse_error)
                    logger.warning(f"Compiler high-param submitter (rigor): Retry failed - {error}")
        except Exception as e:
            logger.error(f"Compiler high-param submitter (rigor): Retry request failed - {e}")
        
        # All retries failed
        logger.error(f"Compiler high-param submitter (rigor): JSON validation failed after retry: {error}")
        return None

