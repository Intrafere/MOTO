"""
Compiler validator - validates document edits for coherence, rigor, and placement.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Tuple

from backend.shared.api_client_manager import api_client_manager
from backend.shared.models import CompilerSubmission, CompilerValidationResult
from backend.shared.json_parser import parse_json
from backend.aggregator.validation.json_validator import json_validator

logger = logging.getLogger(__name__)


def _diagnostic_char_info(text: str, max_chars: int = 100) -> str:
    """
    Generate diagnostic information about a string's characters.
    Used for debugging old_string matching failures.
    """
    if not text:
        return "[EMPTY STRING]"
    
    info_parts = []
    info_parts.append(f"length={len(text)}")
    
    # Check for special characters
    special_chars = []
    for i, char in enumerate(text[:max_chars]):
        code = ord(char)
        # Flag non-ASCII printable characters
        if code < 32 and code not in (9, 10, 13):  # Allow tab, newline, carriage return
            special_chars.append(f"pos{i}:0x{code:02X}({repr(char)})")
        elif code > 127:
            special_chars.append(f"pos{i}:U+{code:04X}({repr(char)})")
    
    if special_chars:
        info_parts.append(f"special_chars=[{', '.join(special_chars[:10])}{'...' if len(special_chars) > 10 else ''}]")
    
    # Line ending analysis
    crlf_count = text.count('\r\n')
    lf_only_count = text.count('\n') - crlf_count
    cr_only_count = text.count('\r') - crlf_count
    if crlf_count or cr_only_count:
        info_parts.append(f"line_endings(CRLF={crlf_count}, LF={lf_only_count}, CR={cr_only_count})")
    
    # Whitespace analysis
    tab_count = text.count('\t')
    double_space_count = text.count('  ')
    if tab_count:
        info_parts.append(f"tabs={tab_count}")
    if double_space_count:
        info_parts.append(f"double_spaces={double_space_count}")
    
    # First/last chars (repr to show escapes)
    first_20 = repr(text[:20])
    last_20 = repr(text[-20:]) if len(text) > 40 else ""
    info_parts.append(f"first20={first_20}")
    if last_20:
        info_parts.append(f"last20={last_20}")
    
    return " | ".join(info_parts)


def normalize_unicode_hyphens(text: str) -> str:
    """
    Normalize Unicode hyphen/dash variants to ASCII hyphen-minus.
    
    This fixes the issue where LLMs output ASCII hyphens but documents
    contain Unicode dashes (en-dash, em-dash, non-breaking hyphen, etc.).
    
    Unicode dash/hyphen characters normalized:
    - U+2010 HYPHEN
    - U+2011 NON-BREAKING HYPHEN  
    - U+2012 FIGURE DASH
    - U+2013 EN DASH
    - U+2014 EM DASH
    - U+2015 HORIZONTAL BAR
    - U+2212 MINUS SIGN
    - U+FE58 SMALL EM DASH
    - U+FE63 SMALL HYPHEN-MINUS
    - U+FF0D FULLWIDTH HYPHEN-MINUS
    """
    if not text:
        return text
    
    # Map of Unicode dashes to ASCII hyphen-minus (U+002D)
    dash_chars = '\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d'
    ascii_hyphen = '-'
    
    result = text
    for dash in dash_chars:
        result = result.replace(dash, ascii_hyphen)
    
    return result


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace to handle LaTeX double-spacing convention.
    
    LaTeX academic writing uses double spaces after sentence-ending periods.
    This function collapses multiple consecutive spaces to single space
    while preserving newlines.
    
    This fixes the issue where LLMs output single-spaced text but documents
    contain double-spaced academic formatting.
    """
    if not text:
        return text
    
    import re
    # Collapse 2+ consecutive spaces to single space
    # Pattern: '  +' matches 2 or more spaces
    return re.sub(r'  +', ' ', text)


def find_with_normalized_hyphens(needle: str, haystack: str) -> Tuple[int, str]:
    """
    Find needle in haystack, using Unicode hyphen normalization if exact match fails.
    
    Returns:
        Tuple of (position, actual_text_in_haystack)
        - position: Index where found (-1 if not found)
        - actual_text_in_haystack: The actual text from haystack that matched
          (may have different Unicode characters than needle)
    """
    # First try exact match
    pos = haystack.find(needle)
    if pos >= 0:
        logger.debug(f"EXACT_MATCH_SUCCESS: old_string found at position {pos}")
        return (pos, needle)
    
    # === DIAGNOSTIC LOGGING FOR FAILED EXACT MATCH ===
    logger.warning(f"EXACT_MATCH_FAILED - Starting diagnostics...")
    logger.warning(f"   NEEDLE: {_diagnostic_char_info(needle)}")
    logger.warning(f"   HAYSTACK: {_diagnostic_char_info(haystack, max_chars=50)}")
    
    # Normalize both and try again
    normalized_needle = normalize_unicode_hyphens(needle)
    normalized_haystack = normalize_unicode_hyphens(haystack)
    
    pos = normalized_haystack.find(normalized_needle)
    if pos >= 0:
        # Found with normalization - extract actual text from original haystack
        # The length should be the same since we only replace 1-char with 1-char
        actual_text = haystack[pos:pos + len(needle)]
        logger.info(f"HYPHEN_NORMALIZED_MATCH: Found at pos {pos}")
        logger.debug(f"   Unicode hyphen normalization matched: '{needle[:50]}...' found as '{actual_text[:50]}...'")
        return (pos, actual_text)
    
    # Try whitespace normalization (3rd layer - handles LaTeX double-spacing)
    ws_needle = normalize_whitespace(normalized_needle)
    ws_haystack = normalize_whitespace(normalized_haystack)
    
    ws_pos = ws_haystack.find(ws_needle)
    if ws_pos >= 0:
        # Found with whitespace normalization - use regex to find actual occurrence in original
        # Convert normalized needle to pattern that allows flexible whitespace
        import re
        escaped = re.escape(ws_needle)
        # Replace single spaces with pattern that matches 1+ whitespace chars
        flexible_pattern = escaped.replace(r'\ ', r'\s+')
        
        match = re.search(flexible_pattern, haystack)
        if match:
            actual_text = match.group(0)
            logger.info(f"WHITESPACE_NORMALIZED_MATCH: Found at pos {match.start()}")
            logger.debug(f"   Whitespace normalization matched: '{needle[:50]}...' found as '{actual_text[:50]}...'")
            return (match.start(), actual_text)
    
    # === DEEP DIAGNOSTICS FOR COMPLETE FAILURE ===
    logger.warning(f"MATCH_FAILED_COMPLETELY - Deep diagnostic analysis:")
    
    # Check for whitespace-only differences
    needle_no_ws = ''.join(needle.split())
    haystack_no_ws = ''.join(haystack.split())
    if needle_no_ws in haystack_no_ws:
        logger.warning(f"   WHITESPACE_DIFF_DETECTED: Match found when ALL whitespace stripped!")
        # Try to find approximate location
        ws_pos = haystack_no_ws.find(needle_no_ws)
        logger.warning(f"   Whitespace-stripped match at pos {ws_pos} (in stripped version)")
        
        # Try to show actual difference
        # Find first divergence point
        for i, (n_char, h_char) in enumerate(zip(needle[:200], haystack[:200])):
            if n_char != h_char:
                logger.warning(f"   First char diff at pos {i}: needle=0x{ord(n_char):02X}({repr(n_char)}) vs haystack=0x{ord(h_char):02X}({repr(h_char)})")
                # Show context around divergence
                context_start = max(0, i - 10)
                context_end = min(len(needle), i + 10)
                logger.warning(f"   Needle around diff: {repr(needle[context_start:context_end])}")
                if i < len(haystack):
                    context_end_h = min(len(haystack), i + 10)
                    logger.warning(f"   Haystack around diff: {repr(haystack[context_start:context_end_h])}")
                break
    
    # Check for line ending differences specifically
    needle_lf = needle.replace('\r\n', '\n').replace('\r', '\n')
    haystack_lf = haystack.replace('\r\n', '\n').replace('\r', '\n')
    if needle_lf in haystack_lf:
        logger.warning(f"   LINE_ENDING_DIFF_DETECTED: Match found when line endings normalized to LF!")
    
    # Check for leading/trailing whitespace differences
    needle_stripped = needle.strip()
    if needle_stripped != needle:
        if needle_stripped in haystack:
            logger.warning(f"   LEADING_TRAILING_WS_DIFF: Match found when needle stripped! needle has {len(needle) - len(needle_stripped)} extra ws chars")
    
    return (-1, "")


class JSONParseError(Exception):
    """Raised when JSON parsing fails - signals need for retry"""
    def __init__(self, message: str, response: str, parse_error: Exception):
        self.message = message
        self.response = response
        self.parse_error = parse_error
        super().__init__(message)


class CompilerValidator:
    """
    Validates compiler submissions.
    - Checks coherence (grammatical + holistic)
    - Checks rigor maintenance
    - Checks placement context validity
    - Checks non-redundancy
    """
    
    def __init__(self, model_name: str, user_prompt: str, websocket_broadcaster: Optional[Callable] = None):
        self.model_name = model_name
        self.user_prompt = user_prompt
        self.websocket_broadcaster = websocket_broadcaster
        self._initialized = False
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "compiler_validator"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"comp_val_{self.task_sequence:03d}"
    
    async def initialize(self) -> None:
        """Initialize validator."""
        if self._initialized:
            return
        
        self._initialized = True
        logger.info(f"Compiler validator initialized with model: {self.model_name}")
    
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
                temperature=0.0,  # Deterministic generation - evolving context provides diversity
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
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response using central parse_json utility.
        Raises JSONParseError on failure to signal retry needed.
        
        The central parse_json() handles:
        - Reasoning tokens (<think>...</think>)
        - Markdown code blocks (```json ... ```)
        - Control tokens
        - LaTeX escape sequences
        - Malformed JSON detection
        - Enhanced error logging
        """
        try:
            # Use central parse_json for consistent sanitization and parsing
            return parse_json(response)
        except (json.JSONDecodeError, ValueError) as e:
            # Raise custom exception to signal retry needed
            logger.error(f"Compiler validator: JSON parse failed - {e}")
            raise JSONParseError(f"Failed to parse JSON response", response, e)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """
        Fallback parser for when all JSON parse attempts fail.
        Uses keyword heuristics to extract decision from natural language.
        
        Strategy: Look for "accept" vs "reject" keywords in response.
        """
        response_lower = response.lower()
        
        # Detect decision via keywords
        decision = "reject"  # Default to reject for safety
        if 'accepted: true' in response_lower or 'decision: accept' in response_lower:
            decision = "accept"
        elif 'accepted: false' in response_lower or 'decision: reject' in response_lower:
            decision = "reject"
        elif 'accept' in response_lower and 'reject' not in response_lower:
            decision = "accept"
        elif 'reject' in response_lower:
            decision = "reject"
        
        logger.warning(f"CompilerValidator: Fallback parser used (determined decision={decision} from keywords)")
        
        return {
            'decision': decision,
            'reasoning': response,  # Full response, no truncation
        }
    
    async def _parse_json_with_retry(
        self, 
        response: str, 
        original_prompt: str,
        system_prompt: str,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Parse JSON with a single conversational retry on failure.
        
        Uses api_client_manager for retry calls to ensure boost/fallback work correctly.
        Only ONE retry is attempted to prevent cascading failures when the coordinator
        also has retry logic at the phase level.
        
        Flow:
        1. Attempt parse via parse_json()
        2. If fails: ask LLM to reformat via conversation (single attempt)
        3. If still fails: use _fallback_parse()
        
        Args:
            response: LLM response to parse
            original_prompt: Original prompt that generated response
            system_prompt: System prompt used (for context, not actively used in retry)
            retry_count: Current retry attempt (0-indexed) - kept for API compatibility
        
        Returns:
            Parsed dict (from JSON or fallback)
        """
        # First attempt: try to parse JSON directly
        try:
            parsed = parse_json(response)
            return parsed
            
        except Exception as parse_error:
            logger.warning(f"CompilerValidator: JSON parse failed, attempting single retry: {parse_error}")
            
            # If we're already in a retry, use fallback immediately to prevent deep recursion
            if retry_count > 0:
                logger.info("CompilerValidator: Already in retry, using fallback parser")
                return self._fallback_parse(response)
            
            # Build retry prompt asking for reformatted JSON
            # Note: response is already truncated to 2000 chars in the prompt text
            reparse_prompt = (
                "Your previous response could not be parsed as valid JSON.\n\n"
                f"YOUR PREVIOUS RESPONSE:\n{response[:2000]}{'...' if len(response) > 2000 else ''}\n\n"
                f"PARSE ERROR: {str(parse_error)}\n\n"
                "Please provide the exact same validation decision in valid JSON format.\n"
                "CRITICAL: Properly escape backslashes (use \\\\) and quotes (use \\\").\n"
                "Respond with ONLY the corrected JSON, no explanation."
            )
            
            # Single retry using api_client_manager (supports boost/fallback)
            try:
                retry_task_id = f"{self.get_current_task_id()}_retry"
                
                # CRITICAL FIX: Truncate failed output to prevent context overflow during retry
                max_failed_output_chars = 2000  # ~500 tokens - enough for error context
                if len(response) > max_failed_output_chars:
                    failed_output_preview = response[:max_failed_output_chars] + "\n[...output truncated for retry...]"
                else:
                    failed_output_preview = response
                
                # Calculate if conversation fits in context window
                from backend.shared.config import system_config, rag_config
                prompt_tokens = count_tokens(original_prompt)
                preview_tokens = count_tokens(failed_output_preview)
                retry_prompt_tokens = count_tokens(reparse_prompt)
                conversation_tokens = prompt_tokens + preview_tokens + retry_prompt_tokens
                max_input = rag_config.get_available_input_tokens(system_config.compiler_validator_context_window, system_config.compiler_validator_max_output_tokens)
                
                if conversation_tokens > max_input:
                    # Too large - just retry with original prompt
                    logger.warning(
                        f"CompilerValidator: Retry conversation too large ({conversation_tokens} > {max_input}), "
                        f"using simple retry without conversation context"
                    )
                    retry_response = await api_client_manager.generate_completion(
                        task_id=retry_task_id,
                        role_id=self.role_id,
                        model=self.model_name,
                        messages=[{"role": "user", "content": original_prompt}],
                        temperature=0.0,
                        max_tokens=system_config.compiler_validator_max_output_tokens
                    )
                else:
                    # Build conversation with truncated failed output
                    retry_response = await api_client_manager.generate_completion(
                        task_id=retry_task_id,
                        role_id=self.role_id,
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": original_prompt},
                            {"role": "assistant", "content": failed_output_preview},
                            {"role": "user", "content": reparse_prompt}
                        ],
                        temperature=0.0,  # Deterministic JSON formatting
                        max_tokens=system_config.compiler_validator_max_output_tokens  # Respect max_tokens on retry
                    )
                
                if not retry_response.get("choices"):
                    logger.error("CompilerValidator: Retry request returned no choices")
                    return self._fallback_parse(response)
                
                message = retry_response["choices"][0]["message"]
                retry_output = message.get("content", "") or message.get("reasoning", "")
                
                try:
                    parsed = parse_json(retry_output)
                    logger.info("CompilerValidator: Retry succeeded!")
                    return parsed
                except Exception as retry_parse_error:
                    logger.warning(f"CompilerValidator: Retry parse failed: {retry_parse_error}")
                    return self._fallback_parse(response)
                    
            except Exception as retry_error:
                logger.error(f"CompilerValidator: Retry request failed - {retry_error}")
                return self._fallback_parse(response)
    
    def _pre_validate_exact_string_match(
        self,
        submission: CompilerSubmission,
        current_paper: str,
        current_outline: Optional[str] = None
    ) -> Optional[CompilerValidationResult]:
        """
        Pre-validate exact string matching BEFORE calling LLM validator.
        
        This separates placement failures (old_string not found) from coherence failures.
        Returns None if pre-validation passes, or a rejection result if it fails.
        
        This is critical because:
        1. The LLM validator often conflates "string not found" with "incoherent content"
        2. Pre-validation provides immediate, specific feedback
        3. Saves an LLM API call when string matching will obviously fail
        
        For outline_update mode, checks against the outline instead of the paper.
        """
        # === DIAGNOSTIC LOGGING FOR PRE-VALIDATION START ===
        logger.info(f"PRE_VALIDATE_START: mode={submission.mode}, operation={submission.operation}")
        if submission.old_string:
            logger.info(f"   old_string preview: {repr(submission.old_string[:100])}{'...' if len(submission.old_string) > 100 else ''}")
            logger.debug(f"   old_string diagnostics: {_diagnostic_char_info(submission.old_string)}")
        
        # Determine which document to check against based on mode
        if submission.mode == "outline_update":
            # For outline_update, check against the outline
            if not current_outline:
                # Can't pre-validate outline_update without the outline - skip pre-validation
                logger.debug("Skipping pre-validation for outline_update: no outline provided")
                return None
            document_to_check = current_outline
            document_name = "outline"
        else:
            # For all other modes (construction, review, rigor), check against the paper
            document_to_check = current_paper
            document_name = "document"
        
        logger.debug(f"   {document_name} length: {len(document_to_check) if document_to_check else 0} chars")
        
        # CRITICAL: Check for empty document FIRST
        # When document is empty, only full_content operation is valid
        document_is_empty = not document_to_check or not document_to_check.strip()
        
        if document_is_empty and submission.operation in ("replace", "insert_after", "delete"):
            logger.warning(f"Pre-validation failed: {document_name.capitalize()} is EMPTY but operation='{submission.operation}' requires existing content")
            return CompilerValidationResult(
                submission_id=submission.submission_id,
                decision="reject",
                reasoning=(
                    f"EMPTY_{document_name.upper()}_ERROR: The {document_name} is currently EMPTY (0 chars). "
                    f"You used operation='{submission.operation}', but this operation requires existing text to match against.\n\n"
                    f"WHEN {document_name.upper()} IS EMPTY, YOU MUST USE operation='full_content'\n\n"
                    f"CORRECT JSON FORMAT FOR EMPTY {document_name.upper()}:\n"
                    f'{{\n'
                    f'  "needs_{"update" if submission.mode == "outline_update" else "construction"}": true,\n'
                    f'  "operation": "full_content",\n'
                    f'  "old_string": "",\n'
                    f'  "new_string": "Your actual content here...",\n'
                    f'  "reasoning": "Writing content using full_content since {document_name} is empty"\n'
                    f'}}\n\n'
                    f"FIX REQUIRED:\n"
                    f"1. Change operation from '{submission.operation}' to 'full_content'\n"
                    f"2. Set old_string to empty string \"\"\n"
                    f"3. Put your content in new_string"
                ),
                summary=f"{document_name.capitalize()} is empty - must use full_content operation, not {submission.operation} (pre-validation)",
                coherence_check=True,   # Not a coherence issue
                rigor_check=True,       # Not a rigor issue
                placement_check=False,  # THIS is the issue - wrong operation for empty document
                json_valid=True
            )
        
        # Only check operations that require old_string matching
        if submission.operation not in ("replace", "insert_after", "delete"):
            return None  # full_content doesn't need old_string
        
        # If no old_string provided for operations that need it
        if not submission.old_string:
            logger.warning(f"Pre-validation failed: {submission.operation} operation requires old_string but none provided")
            return CompilerValidationResult(
                submission_id=submission.submission_id,
                decision="reject",
                reasoning=f"EXACT_MATCH_FAILURE: The '{submission.operation}' operation requires an 'old_string' field specifying the exact text to find in the {document_name}, but no old_string was provided.\n\nFIX REQUIRED:\n1. Provide the exact text you want to match in the 'old_string' field\n2. Include enough context (3-5 lines) to ensure unique matching\n3. The text must exist verbatim in the current {document_name}",
                summary=f"Missing old_string for {submission.operation} operation (pre-validation)",
                coherence_check=True,  # Not a coherence issue
                rigor_check=True,       # Not a rigor issue
                placement_check=False,  # THIS is the issue
                json_valid=True
            )
        
        # Check if old_string exists in the document (with Unicode hyphen normalization)
        pos, actual_text = find_with_normalized_hyphens(submission.old_string, document_to_check)
        
        if pos < 0:
            # NEW: Check if the not-found text exists in the OUTLINE
            # This is a common mistake - models confuse outline content with paper content
            outline_confusion = False
            if current_outline and submission.mode != "outline_update":
                outline_pos, _ = find_with_normalized_hyphens(submission.old_string, current_outline)
                if outline_pos >= 0:
                    outline_confusion = True
            
            if outline_confusion:
                # Provide targeted feedback for outline vs paper confusion
                logger.warning(f"Pre-validation failed: old_string found in OUTLINE but not in PAPER (outline confusion)")
                return CompilerValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=(
                        f"OUTLINE_VS_PAPER_CONFUSION: The text you referenced exists in the OUTLINE, not in the PAPER.\n\n"
                        f"YOUR old_string:\n'{submission.old_string[:200]}...'\n\n"
                        f"This text exists in CURRENT OUTLINE but has NOT been written to the PAPER yet.\n\n"
                        f"FIX REQUIRED:\n"
                        f"1. Review CURRENT DOCUMENT PROGRESS (the actual paper content)\n"
                        f"2. Use old_string from the PAPER, not from the OUTLINE\n"
                        f"3. If you want to write this outline section, use operation='insert_after' with text that EXISTS in the paper as your old_string anchor"
                    ),
                    summary="old_string found in OUTLINE not PAPER - use text from CURRENT DOCUMENT PROGRESS (pre-validation)",
                    coherence_check=True,
                    rigor_check=True,
                    placement_check=False,
                    json_valid=True
                )
            
            # Try to find similar text to provide helpful feedback
            similar_text = self._find_similar_text(submission.old_string, document_to_check)
            
            if similar_text:
                fix_suggestion = f"\n\nSIMILAR TEXT FOUND IN {document_name.upper()}:\n'{similar_text[:300]}...'\n\nFIX: Use the exact text from the {document_name} above."
            else:
                fix_suggestion = f"\n\nNo similar text found. Verify the old_string matches something in the current {document_name} exactly."
            
            logger.warning(f"Pre-validation failed: old_string not found in {document_name}")
            return CompilerValidationResult(
                submission_id=submission.submission_id,
                decision="reject",
                reasoning=f"EXACT_MATCH_FAILURE: The old_string provided does not exist in the current {document_name}.\n\nOLD_STRING PROVIDED:\n'{submission.old_string[:200]}...'\n\nTHIS TEXT WAS NOT FOUND in the current {document_name}. The exact string matching system requires the old_string to exist VERBATIM.{fix_suggestion}",
                summary=f"old_string not found in {document_name} (pre-validation)",
                coherence_check=True,   # Not a coherence issue
                rigor_check=True,        # Not a rigor issue  
                placement_check=False,   # THIS is the issue
                json_valid=True
            )
        
        # If Unicode normalization was used, update the submission's old_string to the actual text
        # This ensures the coordinator can apply the edit correctly
        if actual_text != submission.old_string:
            logger.info(f"Unicode hyphen normalization applied - updating old_string to actual document text")
            submission.old_string = actual_text
        
        # Check if old_string is unique (only appears once) - use normalized comparison
        normalized_doc = normalize_unicode_hyphens(document_to_check)
        normalized_old = normalize_unicode_hyphens(submission.old_string)
        match_count = normalized_doc.count(normalized_old)
        if match_count > 1:
            logger.warning(f"Pre-validation failed: old_string appears {match_count} times in {document_name} (not unique)")
            return CompilerValidationResult(
                submission_id=submission.submission_id,
                decision="reject",
                reasoning=f"EXACT_MATCH_FAILURE: The old_string appears {match_count} times in the {document_name}. It must be UNIQUE (appear exactly once).\n\nOLD_STRING:\n'{submission.old_string[:200]}...'\n\nFIX REQUIRED:\nInclude more surrounding context to make the match unique. For example:\n- Include the preceding paragraph or sentence\n- Include the section header above the target text\n- Use 3-5 lines of context instead of 1-2",
                summary=f"old_string matches {match_count} locations in {document_name}, not unique (pre-validation)",
                coherence_check=True,   # Not a coherence issue
                rigor_check=True,        # Not a rigor issue
                placement_check=False,   # THIS is the issue
                json_valid=True
            )
        
        # Pre-validation passed
        logger.info(f"PRE_VALIDATE_SUCCESS: old_string found uniquely at position {pos}")
        return None
    
    def _find_similar_text(self, target: str, document: str, threshold: float = 0.6) -> Optional[str]:
        """
        Find text in document that's similar to target (for helpful error messages).
        Uses simple substring matching for efficiency.
        """
        # Try to find the first few words of target in the document
        if not target or not document:
            return None
        
        # Get first 50 chars of target as search key
        search_key = target[:50].strip()
        if len(search_key) < 10:
            return None
        
        # Try finding partial matches with first/last portions
        first_words = ' '.join(search_key.split()[:5])
        
        # Search for first few words (exact match first)
        if first_words and len(first_words) > 10:
            idx = document.find(first_words)
            if idx >= 0:
                # Found partial match - return surrounding context
                start = max(0, idx - 20)
                end = min(len(document), idx + len(target) + 50)
                return document[start:end].strip()
            
            # Try with Unicode hyphen normalization
            normalized_first_words = normalize_unicode_hyphens(first_words)
            normalized_document = normalize_unicode_hyphens(document)
            idx = normalized_document.find(normalized_first_words)
            if idx >= 0:
                # Found with normalization - return from original document
                start = max(0, idx - 20)
                end = min(len(document), idx + len(target) + 50)
                return document[start:end].strip()
        
        return None

    async def validate_submission(
        self,
        submission: CompilerSubmission,
        current_paper: str,
        current_outline: Optional[str] = None
    ) -> CompilerValidationResult:
        """
        Validate a compiler submission.
        
        Args:
            submission: Submission to validate
            current_paper: Current document content
            current_outline: Current outline (if applicable)
        
        Returns:
            CompilerValidationResult
        """
        logger.info(f"Validating {submission.mode} submission: {submission.submission_id}")
        
        # PRE-PROCESSING: Strip any placeholder text from submission content and new_string
        # Instead of rejecting, we silently strip placeholders to simplify the workflow
        original_content = submission.content
        original_new_string = submission.new_string
        
        stripped_content = self._strip_placeholder_text(submission.content) if submission.content else submission.content
        stripped_new_string = self._strip_placeholder_text(submission.new_string) if submission.new_string else submission.new_string
        
        # Check if we stripped anything (for logging purposes)
        if stripped_content != original_content or stripped_new_string != original_new_string:
            logger.info(f"Stripped placeholder text from submission (content: {len(original_content or '')} -> {len(stripped_content or '')} chars, new_string: {len(original_new_string or '')} -> {len(stripped_new_string or '')} chars)")
            # Update the submission with stripped content
            submission.content = stripped_content
            submission.new_string = stripped_new_string
        
        # PRE-VALIDATION: Check exact string matching BEFORE calling LLM
        # This separates placement failures from coherence failures
        # For outline_update mode, this checks against the outline; otherwise against the paper
        string_match_result = self._pre_validate_exact_string_match(submission, current_paper, current_outline)
        if string_match_result is not None:
            logger.info(f"Pre-validation rejected: {string_match_result.summary}")
            return string_match_result
        
        # Build validation prompt
        prompt = self._build_validation_prompt(submission, current_paper, current_outline)
        
        # CRITICAL: Verify actual prompt size fits in context window
        from backend.shared.utils import count_tokens
        from backend.shared.config import system_config, rag_config
        actual_prompt_tokens = count_tokens(prompt)
        max_allowed_tokens = rag_config.get_available_input_tokens(system_config.compiler_validator_context_window, system_config.compiler_validator_max_output_tokens)
        
        if actual_prompt_tokens > max_allowed_tokens:
            logger.error(
                f"Compiler validator: Assembled prompt ({actual_prompt_tokens} tokens) exceeds context window "
                f"({max_allowed_tokens} tokens after safety margin). This indicates a context allocation bug."
            )
            return CompilerValidationResult(
                submission_id=submission.submission_id,
                decision="reject",
                reasoning=f"Internal error: Prompt too large ({actual_prompt_tokens} tokens > {max_allowed_tokens} max)",
                summary="Internal context overflow error",
                json_valid=False
            )
        
        logger.debug(f"Compiler validator prompt: {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
        
        # Generate task ID for tracking
        task_id = self.get_current_task_id()
        self.task_sequence += 1
        
        # Notify task started (for workflow panel)
        if self.task_tracking_callback:
            self.task_tracking_callback("started", task_id)
        
        try:
            # Get validation from LM with truncation retry via api_client_manager (handles boost and fallback)
            logger.info(f"Generating validation via api_client_manager (task_id={task_id})...")
            response = await self._generate_with_truncation_retry(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic validation - evolving context provides diversity
                operation_name=f"validate_{submission.mode}",
                task_id=task_id,
                max_tokens_override=system_config.compiler_validator_max_output_tokens  # User-configurable
            )
            
            # Extract content from either 'content' or 'reasoning' field
            # Some reasoning models (e.g., DeepSeek R1, certain GPT variants) output JSON in 'reasoning' field
            message = response["choices"][0]["message"]
            llm_output = message.get("content", "") or message.get("reasoning", "")
            
            # Parse JSON response with retry logic
            validation_data = await self._parse_json_with_retry(
                llm_output,
                prompt,
                "",  # system_prompt not used in this call
                0  # initial retry count
            )
            
            # Extract validation decision
            decision = validation_data.get("decision", "reject")
            reasoning = validation_data.get("reasoning", "No reasoning provided")
            coherence = validation_data.get("coherence_check", False)
            rigor = validation_data.get("rigor_check", False)
            placement = validation_data.get("placement_check", False)
            
            # Create summary for rejection log (max 750 chars)
            summary = reasoning[:750]
            
            result = CompilerValidationResult(
                submission_id=submission.submission_id,
                decision=decision,
                reasoning=reasoning,
                summary=summary,
                coherence_check=coherence,
                rigor_check=rigor,
                placement_check=placement,
                json_valid=True
            )
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"Validation result: {decision} (coherence={coherence}, rigor={rigor}, placement={placement})")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            # Notify task completed (failed but still completed)
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            return CompilerValidationResult(
                submission_id=submission.submission_id,
                decision="reject",
                reasoning=f"Validation error: {str(e)}",
                summary=f"Validation error: {str(e)}"[:750],
                json_valid=False
            )
    
    def _strip_placeholder_text(self, text: str) -> str:
        """
        Strip any placeholder markers from text.
        Placeholders are system-managed markers that should not appear in submissions.
        Instead of rejecting, we silently strip them to simplify the workflow.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with all placeholder markers removed
        """
        if not text:
            return text
        
        # Import the exact placeholder constants from paper_memory
        from backend.compiler.memory.paper_memory import (
            ABSTRACT_PLACEHOLDER,
            INTRO_PLACEHOLDER,
            CONCLUSION_PLACEHOLDER,
            PAPER_ANCHOR
        )
        
        # List of exact placeholder strings to strip
        placeholders_to_strip = [
            ABSTRACT_PLACEHOLDER,
            INTRO_PLACEHOLDER,
            CONCLUSION_PLACEHOLDER,
            PAPER_ANCHOR
        ]
        
        result = text
        for placeholder in placeholders_to_strip:
            result = result.replace(placeholder, "")
        
        # Also strip any generic placeholder patterns that might be variations
        # These are the detection keywords that were previously used for rejection
        import re
        # Pattern for any [HARD CODED ... ] or [PLACEHOLDER FOR ...] style markers
        result = re.sub(r'\[HARD CODED[^\]]*\]', '', result)
        result = re.sub(r'\[PLACEHOLDER FOR[^\]]*\]', '', result)
        
        # Clean up any resulting double newlines or excessive whitespace
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _build_validation_prompt(
        self,
        submission: CompilerSubmission,
        current_paper: str,
        current_outline: Optional[str]
    ) -> str:
        """Build validation prompt based on submission mode."""
        # Route to appropriate system prompt based on mode
        if submission.mode in ["outline_create", "outline_update"]:
            system_prompt = self._get_outline_validation_system_prompt(submission.mode)
        else:
            system_prompt = self._get_paper_validation_system_prompt(submission.mode)
        
        parts = [
            system_prompt,
            "\n---\n",
            self._get_validation_json_schema(),
            "\n---\n",
            f"USER COMPILER-DIRECTING PROMPT:\n{self.user_prompt}",
            "\n---\n",
        ]
        
        if current_outline:
            parts.append(f"CURRENT OUTLINE:\n{current_outline}\n---\n")
        
        parts.append(f"CURRENT DOCUMENT:\n{current_paper}\n---\n")
        parts.append(f"SUBMISSION TO VALIDATE:\n")
        parts.append(f"Operation: {submission.operation}\n")
        if submission.old_string:
            parts.append(f"Old String (to find): {submission.old_string}\n")
        parts.append(f"New String: {submission.new_string}\n")
        parts.append(f"Content: {submission.content}\n")
        parts.append(f"Reasoning: {submission.reasoning}\n")
        parts.append("\n---\n")
        parts.append("Now validate this submission (respond as JSON):")
        
        return "\n".join(parts)
    
    def _get_outline_validation_system_prompt(self, mode: str) -> str:
        """Get system prompt for OUTLINE validation (outline_create, outline_update)."""
        base_prompt = """You are validating a mathematical document outline submission. Your role is to decide if this submission should be ACCEPTED or REJECTED.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

WEB SEARCH STRONGLY ENCOURAGED:
If your model has access to real-time web search capabilities (such as Perplexity Sonar or similar), you are STRONGLY ENCOURAGED to use them to:
- Verify mathematical claims against current published research
- Access recent developments and contemporary mathematical literature
- Cross-reference theorems, proofs, and techniques with authoritative sources
- Supplement analysis with verified external information
- Validate approaches against established mathematical consensus

The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous, verifiable mathematical content. Use all available resources - internal context as exploration history, your base knowledge for reasoning, and web search (if available) for verification and current information.

WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth. If you have web search, use it.

---

CRITICAL - SYSTEM-MANAGED MARKERS (NOT AI-GENERATED):

The CURRENT OUTLINE may contain hard-coded markers that the SYSTEM adds automatically:

**OUTLINE ANCHORS** (mark outline boundaries):
- [HARD CODED BRACKETED DESIGNATION THAT SHOWS END-OF-PAPER DESIGNATION MARK]
- [HARD CODED END-OF-OUTLINE MARK -- ALL OUTLINE CONTENT SHOULD BE ABOVE THIS LINE]

CRITICAL DISTINCTIONS:
1. **Markers in CURRENT OUTLINE**: SYSTEM-MANAGED and EXPECTED. The code in outline_memory.py adds these automatically. They are NOT AI-generated content.

2. **Markers in SUBMISSION CONTENT**: ABSOLUTELY FORBIDDEN. Submissions must NEVER contain these markers. Pre-validation check (line 326) catches this before you see the submission.

WHY ANCHORS EXIST: They mark outline boundaries and prevent content from being added beyond the intended endpoint.

FOR CRITERION #11 (NO PLACEHOLDER TEXT): This checks the SUBMISSION ONLY, not the current outline. Anchor markers in the current outline are expected and normal. DO NOT reject just because the current outline contains anchor markers.

---

REQUIRED SECTION STRUCTURE (MANDATORY):
Every outline MUST include these exact sections with these exact names in this exact order:
1. **Abstract** - Must be named exactly "Abstract" (appears first)
2. **Introduction** - Must be named exactly "Introduction" or "I. Introduction" (after Abstract)
3. **Body Sections** - At least one body section (II, III, IV, etc.) between Introduction and Conclusion
4. **Conclusion** - Must be named exactly "Conclusion" or "N. Conclusion" (always LAST content section)

OUTLINE VALIDATION CRITERIA:
1. SECTION_STRUCTURE: MUST include Abstract, Introduction, at least one Body section, and Conclusion with exact names
2. SECTION_ORDER: Abstract → Introduction → Body sections → Conclusion (this exact order)
3. COHERENCE: Logically structured with clear sections and subsections
4. COMPLETENESS: Captures all relevant content from aggregator database in relation to what is relevant to the title set in the user prompt
5. ALIGNMENT: Aligns with user's compiler-directing prompt goals
6. COMPREHENSIVENESS: Provides sufficient detail to guide mathematical document construction
7. MATHEMATICAL PROGRESSION: Body sections follow logical progression (definitions → main results → theorems → proofs)
8. IN-TEXT CITATIONS ONLY: Must NOT include a separate References or Citations section
9. ANCHOR PRESERVATION: Must not attempt to add content after the end-of-outline anchor markers
10. LOGICAL GROUNDING: Outline must reference actual content from aggregator database based on sound mathematical principles, not unfounded claims
11. NO PLACEHOLDER TEXT: Must not contain any placeholder markers (e.g., "[HARD CODED PLACEHOLDER FOR...", "[PLACEHOLDER FOR...", "TO BE WRITTEN AFTER..."). Placeholders are structural markers only - all submitted content must be actual outline content.

**CRITICAL**: Criterion #11 checks the SUBMISSION CONTENT ONLY. The CURRENT OUTLINE may contain system-managed anchor markers (normal and expected). Do NOT reject a submission just because anchor markers exist in the current outline - only reject if the SUBMISSION ITSELF contains placeholder or anchor text (pre-validation at line 326 catches this before you see it).

YOUR TASK:
Verify the submission meets ALL criteria above. Accept only if ALL criteria pass. Reject if ANY criterion fails.

REJECTION CATEGORIES (provide specific feedback):
- MISSING_REQUIRED_SECTION: Missing Abstract, Introduction, or Conclusion (must have all three with exact names)
- INCORRECT_SECTION_ORDER: Sections are out of order (must be: Abstract → Introduction → Body → Conclusion)
- INCORRECT_SECTION_NAME: Section names don't match exactly (e.g., "Summary" instead of "Conclusion", "Overview" instead of "Introduction")
- STRUCTURAL: Body sections not in logical mathematical progression order
- INCOMPLETENESS: Missing critical content from aggregator database that's relevant to document title
- MISALIGNMENT: Doesn't serve user's compiler-directing prompt goals
- INSUFFICIENT_DETAIL: Lacks necessary granularity to guide mathematical document construction
- FORMAT_VIOLATION: Includes separate References/Citations section (NOT allowed)
- ANCHOR_VIOLATION: Content placed after outline anchor markers

SECTION NAME VALIDATION (CRITICAL):

You MUST check if the submission content contains these EXACT section headers:

1. **Abstract Header Check**:
   ✓ VALID: A line containing ONLY "Abstract" (case-insensitive, may have whitespace)
   ✓ VALID: "Abstract" or "ABSTRACT" or "  Abstract  "
   ❌ INVALID: "Summary of the paper" or "Abstract: This paper" or any descriptive text instead of just "Abstract"
   
2. **Introduction Header Check**:
   ✓ VALID: "Introduction" or "I. Introduction" (case-insensitive)
   ✓ VALID: "INTRODUCTION" or "I. INTRODUCTION"
   ❌ INVALID: "Overview" or "Background"

3. **Conclusion Header Check**:
   ✓ VALID: "Conclusion" or "V. Conclusion" (with any Roman numeral, case-insensitive)
   ✓ VALID: "CONCLUSION" or "VII. CONCLUSION"
   ❌ INVALID: "Summary" or "Final Remarks"

REJECTION FORMAT - If ANY header is missing or incorrect, reject with this EXACT format:

"MISSING_REQUIRED_SECTION: [Abstract|Introduction|Conclusion]

WHAT I SAW IN YOUR SUBMISSION:
Line 1: '[actual first line of submission]'

WHAT I EXPECTED:
Line 1: 'Abstract'

FIX REQUIRED:
Change your first line from the descriptive text to ONLY the word 'Abstract'.

This is an OUTLINE showing section names - not the actual paper content. Use just 'Abstract' as a header, not a description of what the abstract will contain."

"""
        
        mode_specific = {
            "outline_create": """MODE-SPECIFIC CRITERIA (Outline Creation):
- Outline MUST include: Abstract, Introduction, at least one Body section, Conclusion
- Section names MUST match exactly: "Abstract", "Introduction", "Conclusion"
- Outline captures all relevant unique content from aggregator database
- Outline provides clear structure for mathematical document construction
- Outline aligns with user's compiler-directing prompt
- Body sections follow logical mathematical progression
- Outline is comprehensive enough to guide entire exposition

FEEDBACK FOR ITERATIVE REFINEMENT:
Your feedback will be shown to the submitter for their next iteration. Provide constructive, actionable feedback whether accepting or rejecting.

IF ACCEPTING THE OUTLINE:
- Explain what the outline does well (structure, coverage, alignment)
- Suggest optional improvements the submitter could consider
- Note any minor gaps or refinements that could strengthen the outline
- Remember: acceptance doesn't end refinement - submitter decides when to lock

IF REJECTING THE OUTLINE - USE THIS STRUCTURED FORMAT:

"REJECTION REASON: [Category] - [Specific Issue]

VALIDATION CRITERIA FAILED: [List which criteria failed]

WHAT I SAW IN YOUR SUBMISSION:
[Specific excerpt or description showing the problem]

WHAT I EXPECTED TO SEE:
[Concrete example of correct format or required content]

FIX REQUIRED:
[Step-by-step actionable instructions]

EXAMPLE OF CORRECT FORMAT:
[Complete example of what would be accepted]"

FEEDBACK QUALITY EXAMPLES:

✅ GOOD (Specific and Actionable):
"REJECTION REASON: MISSING_REQUIRED_SECTION - Abstract

VALIDATION CRITERIA FAILED: #1 SECTION_STRUCTURE

WHAT I SAW:
Your outline starts with: 'Summary of the paper's core contribution...'

WHAT I EXPECTED:
First line: 'Abstract' (just this word, no descriptive text)

FIX REQUIRED:
1. Remove the descriptive text from line 1
2. Replace it with ONLY the word 'Abstract'
3. The outline lists section names, not content

EXAMPLE:
Abstract

I. Introduction
   A. Historical context
   ..."

❌ BAD (Vague):
"The outline needs more detail in the main results section."

✅ GOOD (Optional Improvement):
"The outline comprehensively covers all required content. Optional: Consider adding a subsection on worked examples under Section IV to enhance clarity, though not required."

❌ BAD (Not Actionable):
"Missing Abstract section" [Doesn't explain what's wrong or how to fix]

Your feedback should help the submitter produce the best possible outline for guiding paper construction.

ACCEPT if: All required sections present with correct names + all other criteria met
REJECT if: Missing Abstract/Introduction/Conclusion, incorrect section names, or any other criterion fails""",
            
            "outline_update": """MODE-SPECIFIC CRITERIA (Outline Update):
- Update MUST NOT remove or rename Abstract, Introduction, or Conclusion sections
- Update is necessary (missing content or better structure needed)
- Update follows the existing document's already-constructed format and section ordering
- Update is STRICTLY ADDITIVE ONLY - only adds new sections, never modifies or removes existing structure
- New body sections MUST be inserted between Introduction and Conclusion
- Update respects the current document's construction order
- Update maintains logical progression consistency with existing outline structure
- Update doesn't require modification of already-constructed document content
- Changes are substantive, not cosmetic
- New sections maintain mathematical document ordering
- Update does NOT add a References or Citations section

CRITICAL ADDITIVITY CHECK:
- REJECT if update would rename Abstract, Introduction, or Conclusion
- REJECT if update would insert content after Conclusion (except appendix)
- REJECT if update would require editing/moving/renaming already-written document sections
- REJECT if update disrupts the flow of existing document content
- REJECT if update adds a separate References or Citations section
- ACCEPT only if new outline sections can be added without touching what's already written

ACCEPT if: All general criteria + mode-specific criteria met AND update is purely additive
REJECT if: Update is unnecessary, harmful, breaks required section structure, or requires document restructuring"""
        }
        
        return base_prompt + mode_specific.get(mode, mode_specific["outline_create"])
    
    def _get_paper_validation_system_prompt(self, mode: str) -> str:
        """Get system prompt for DOCUMENT validation (construction, review, rigor)."""
        base_prompt = """You are validating a mathematical document construction submission. Your role is to decide if this submission should be ACCEPTED or REJECTED.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

WEB SEARCH STRONGLY ENCOURAGED:
If your model has access to real-time web search capabilities (such as Perplexity Sonar or similar), you are STRONGLY ENCOURAGED to use them to:
- Verify mathematical claims against current published research
- Access recent developments and contemporary mathematical literature
- Cross-reference theorems, proofs, and techniques with authoritative sources
- Supplement analysis with verified external information
- Validate approaches against established mathematical consensus

The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous, verifiable mathematical content. Use all available resources - internal context as exploration history, your base knowledge for reasoning, and web search (if available) for verification and current information.

WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth. If you have web search, use it.

---

CRITICAL - SYSTEM-MANAGED MARKERS (NOT AI-GENERATED):

The CURRENT DOCUMENT may contain hard-coded markers that the SYSTEM adds automatically:

**SECTION PLACEHOLDERS** (show where sections will be written):
- [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION - TO BE WRITTEN AFTER THE INTRODUCTION IS COMPLETE]
- [HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION - TO BE WRITTEN AFTER THE CONCLUSION SECTION IS COMPLETE]
- [HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION - TO BE WRITTEN AFTER THE BODY SECTION IS COMPLETE]

**PAPER ANCHOR** (marks document boundary):
- [HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]

CRITICAL DISTINCTIONS:
1. **Markers in CURRENT DOCUMENT**: SYSTEM-MANAGED and EXPECTED. The code in paper_memory.py adds these automatically. They are NOT AI-generated content.

2. **Markers in SUBMISSION CONTENT**: ABSOLUTELY FORBIDDEN. Submissions must NEVER contain these markers. Pre-validation check (line 326) catches this before you see the submission.

WHY PLACEHOLDERS EXIST: They show where sections will be written and that they DON'T exist yet. When a section is validated, the system REPLACES the placeholder with that content.

FOR CRITERION #11 (NO PLACEHOLDER TEXT): This checks the SUBMISSION ONLY, not the current document. Placeholders in the current document are expected during construction. DO NOT reject just because the current document contains placeholders.

---

CRITICAL DECISION RULE:
If ANY single criterion below fails, you MUST reject the submission. All criteria must pass for acceptance.

DOCUMENT VALIDATION CRITERIA:
1. COHERENCE: Grammatically correct AND maintains holistic document coherence
2. MATHEMATICAL RIGOR: The submission's content is mathematically correct, logically sound, rooted in established mathematical principles (NO unfounded claims or logical fallacies), and is rigorous enough for a mathematical exposition
3. EXACT STRING MATCH VALIDITY: For replace/insert_after/delete operations, the "old_string" MUST exist EXACTLY and UNIQUELY in the current document. For "full_content" operation, no old_string match is required.
4. OPERATION COHERENCE: The operation type must be appropriate (replace for modifications, insert_after for additions, delete for removals, full_content for new sections). Content flows naturally at the specified location, adds imperative content that efficiently contributes towards completion of the document as per the outline, and the content is not overly verbose given the implied scope of the titled document. The content addition adheres to the outline structure for the document.
5. NON-REDUNDANCY: Doesn't repeat existing document content or duplicate any existing sections of the document. The addition strictly adds to the document so the addition does not deviate the document structure from the outline.
6. NO SECTION DUPLICATION: Must not create a section header (e.g., "III. Main Results", "IV. Proofs") OR subsection header (e.g., "V.E.", "IX.C.") that already exists in the current document. Check BOTH the submission content AND the current document for duplicate section headers and subsection headers.
7. NO FORWARD-LOOKING PREVIEWS: Must not contain forward-looking structural previews (e.g., 'Section II will...', 'We will examine...', organized-as-follows lists) UNLESS this is the introduction/first section where a brief roadmap is allowed
8. IN-TEXT CITATIONS ONLY: Must use in-text citations (if any) and NOT create a separate References or Citations section
9. ANCHOR PRESERVATION: Must not attempt to add content after the end-of-document anchor marker ("[HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]")
10. LOGICAL GROUNDING: Must not contain unfounded claims or logical fallacies. All claims must be grounded in established mathematical principles and sound reasoning.
11. NO PLACEHOLDER TEXT: Must not contain any placeholder markers (e.g., "[HARD CODED PLACEHOLDER FOR...", "[PLACEHOLDER FOR...", "TO BE WRITTEN AFTER..."). Placeholders are structural markers indicating where sections WILL BE written - all submitted content must be actual mathematical prose, not placeholder text.

**CRITICAL**: Criterion #11 checks the SUBMISSION CONTENT ONLY. The CURRENT DOCUMENT may contain system-managed placeholders and anchors (normal during construction). Do NOT reject a submission just because placeholders exist in the current document - only reject if the SUBMISSION ITSELF contains placeholder text (pre-validation at line 326 catches this before you see it).

YOUR TASK:
Verify the submission meets ALL criteria above. If even ONE criterion fails, reject the submission.

SECTION DUPLICATION CHECK (CRITICAL):
- The CURRENT OUTLINE is a TEMPLATE showing what sections SHOULD BE written - it is NOT actual content
- The CURRENT PAPER/DOCUMENT is the ACTUAL written content
- Before accepting, scan ONLY the CURRENT PAPER/DOCUMENT (not the outline) for section headers (e.g., "I.", "II.", "III.", "IV.", "V.")
- Check if the SUBMISSION CONTENT contains any section header that already exists in the CURRENT PAPER/DOCUMENT (actual written content)
- DO NOT reject just because a section header appears in the OUTLINE - the outline is a template, not content
- REJECT only if duplicate section headers exist in the ACTUAL CURRENT PAPER/DOCUMENT
- ACCEPT if the submission is filling in a section from the outline that hasn't been written yet in the actual document

EXACT STRING MATCHING VALIDATION (CRITICAL):
The submission uses an exact string matching system for edits:
- "operation" field specifies: "replace", "insert_after", "delete", or "full_content"
- "old_string" field is the EXACT text to find (for replace/insert_after/delete operations)
- "new_string" field is the replacement/insertion text

VALIDATION RULES:
- For "replace": old_string MUST exist EXACTLY (verbatim) in the CURRENT DOCUMENT. Verify the exact match exists.
- For "insert_after": old_string MUST exist EXACTLY in the CURRENT DOCUMENT. Content will be inserted immediately after it.
- For "delete": old_string MUST exist EXACTLY in the CURRENT DOCUMENT. That text will be removed.
- For "full_content": This is for new sections or complete replacements. No old_string needed.
- old_string must be UNIQUE in the document (only one match). If ambiguous, reject.
- Content must flow coherently at the specified location
- Must follow outline structure (sections in logical mathematical progression order)
- Reject if old_string doesn't exist in the current document (for replace/insert_after/delete)
- Reject if operation type doesn't match the intent (e.g., using "replace" when content should be added)

FORWARD-LOOKING PREVIEW DETECTION (CRITICAL):
- Reject if content contains: "Section X will...", "organized as follows:", "we will discuss...", "next we examine...", bullet lists describing future sections
- Brief roadmap acceptable ONLY in introduction/first section; reject everywhere else
- Content should be actual mathematical prose (definitions, theorems, proofs, analysis), not structural previews

CITATION FORMAT CHECK (if applicable):
- Reject if content creates a "References" or "Citations" section header
- Reject if content appears to be a bibliography or reference list
- In-text citations (if present) should be integrated within the narrative

MATHEMATICAL RIGOR CHECK (CRITICAL):
- Reject if content contains unfounded claims or logical fallacies
- Accept only content rooted in established mathematical principles and sound reasoning

"""
        
        mode_specific = {
            "construction": """MODE-SPECIFIC CRITERIA (Document Construction):
- Outline Adherence: Follows the outline structure appropriately
- Logical Flow: Builds logically from existing document content
- Evidence Integration: Captures relevant aggregator database content
- Non-Repetition: Doesn't repeat existing document sections
- Section Uniqueness: Doesn't create section headers that already exist in current document
- Exact String Match Validity: For replace/insert_after/delete, old_string must exist exactly and uniquely
- Operation Appropriateness: Operation type matches the editing intent
- Content Check: No forward-looking structural previews outside introduction
- Mathematical Accuracy: Content is based on established mathematical principles and sound reasoning

VERIFICATION CHECKLIST:
✓ Does it follow the current outline (the outline is the TEMPLATE, not actual content)?
✓ Does it build coherently from what's already written in the ACTUAL DOCUMENT?
✓ Does it integrate content from the aggregator database?
✓ Does it avoid repeating existing content in the ACTUAL DOCUMENT?
✓ Does it avoid creating duplicate section headers that exist in the CURRENT PAPER/DOCUMENT (NOT the outline template)?
✓ For replace/insert_after/delete: Does the old_string exist EXACTLY in the current document?
✓ Is the old_string unique (only appears once) in the document?
✓ Is the operation type appropriate for the intended edit?
✓ Does it avoid forward-looking structural language (unless introduction)?
✓ Is the mathematical content based on established principles and sound reasoning?

REJECTION FEEDBACK FORMAT:
If rejecting, provide CONCRETE, ACTIONABLE feedback using this structure:

"REJECTION REASON: [COHERENCE|RIGOR|EXACT_MATCH|OPERATION|REDUNDANCY|DUPLICATION|etc.]

CRITERION FAILED: [Which specific criterion failed]

ISSUE IDENTIFIED:
[Specific problem in the submission]

FIX REQUIRED:
[Concrete step-by-step instructions]

For EXACT_MATCH failures, show:
- What old_string was provided
- What text actually exists in the document (if similar)
- How to fix it (include more context, fix typo, etc.)"

ACCEPT if: All general criteria + all mode-specific criteria met
REJECT if: Any criterion fails (especially exact string match failures, duplicate section headers, or unsound mathematical claims)""",
            
            "review": """MODE-SPECIFIC CRITERIA (Document Review):
- Edit genuinely improves the document (not cosmetic)
- Edit targets a real issue (grammar, redundancy, coherence, mathematical accuracy)
- Edit maintains or improves overall quality
- Exact String Match Validity: For replace/insert_after/delete, old_string must exist exactly and uniquely
- Operation Appropriateness: Operation type matches the editing intent (replace for corrections, delete for removals)
- Content Check: No forward-looking structural previews outside introduction

REJECTION FEEDBACK FORMAT:
If rejecting, use this structure:
"REJECTION REASON: [Unnecessary|Harmful|Reduces Coherence|EXACT_MATCH|etc.]
ISSUE: [What's wrong]
FIX: [How to improve]"

ACCEPT if: All general criteria + mode-specific criteria met
REJECT if: Edit is unnecessary, harmful, or exact string match fails""",
            
            "rigor": """MODE-SPECIFIC CRITERIA (Mathematical Rigor Enhancement):
- Enhancement genuinely adds mathematical rigor
- Enhancement maintains existing narrative and structure
- Enhancement is appropriate for current draft stage
- Enhancement doesn't reduce clarity
- Exact String Match Validity: For replace/insert_after/delete, old_string must exist exactly and uniquely
- Operation Appropriateness: Operation type matches the enhancement intent
- Content Check: No forward-looking structural previews outside introduction
- Mathematical Check: Enhancement provides rigorous, sound mathematical improvements

REJECTION FEEDBACK FORMAT:
If rejecting, use this structure:
"REJECTION REASON: [Doesn't Add Rigor|Reduces Clarity|Unsound|EXACT_MATCH|etc.]
ISSUE: [What's wrong]
FIX: [What would be acceptable]"

ACCEPT if: All general criteria + mode-specific criteria met
REJECT if: Enhancement doesn't add rigor, reduces quality, introduces unsound mathematical claims, or exact string match fails"""
        }
        
        return base_prompt + mode_specific.get(mode, mode_specific["construction"])
    
    def _get_validation_json_schema(self) -> str:
        """Get JSON schema for validation response."""
        return """
REQUIRED JSON FORMAT:
{
  "decision": "accept" or "reject",
  "reasoning": "string - detailed explanation of decision",
  "coherence_check": boolean - true if coherent,
  "rigor_check": boolean - true if maintains/improves rigor,
  "placement_check": boolean - true if exact string match is valid and operation is appropriate
}

Example (accept):
{
  "decision": "accept",
  "reasoning": "This section follows the outline, builds logically from the definitions, maintains coherent narrative flow. The old_string 'II. Preliminaries\\n\\nIn this section' exists exactly and uniquely in the document, and the insert_after operation is appropriate for adding the new content. No redundancy with existing material.",
  "coherence_check": true,
  "rigor_check": true,
  "placement_check": true
}

Example (reject - old_string not found):
{
  "decision": "reject",
  "reasoning": "The old_string 'II. Preliiminary Section' does not exist in the current document. The actual section header is 'II. Preliminaries'. The submitter should use the exact text from the document.",
  "coherence_check": true,
  "rigor_check": true,
  "placement_check": false
}

Example (reject - ambiguous match):
{
  "decision": "reject",
  "reasoning": "The old_string 'theorem' appears multiple times in the document. The submitter needs to include more surrounding context to make the match unique - for example, include the full theorem statement or preceding sentence.",
  "coherence_check": true,
  "rigor_check": true,
  "placement_check": false
}
"""
    
    def _parse_validation_response(self, response: str) -> Optional[dict]:
        """Parse validation response JSON using centralized validator."""
        try:
            valid, parsed, error = json_validator.validate_compiler_validator_json(response)
            
            if not valid:
                logger.error(f"JSON validation failed for compiler validator: {error}")
                return None
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse validation response: {e}")
            return None
    
    async def validate_rewrite_decision(
        self,
        decision_result: Dict,
        user_prompt: str,
        current_body: str,
        current_outline: str,
        current_title: str,
        critique_feedback: str,
        aggregator_db: str
    ) -> bool:
        """
        Validate a rewrite vs continue decision made after critique phase.
        
        Args:
            decision_result: The decision dict from critique submitter
            user_prompt: User's compiler-directing prompt
            current_body: Body section being evaluated
            current_outline: Paper outline
            current_title: Current paper title
            critique_feedback: All accepted critiques (typically 1-3 out of 5 total attempts)
            aggregator_db: Aggregator database content
            
        Returns:
            True if decision is valid, False if should be retried
        """
        try:
            logger.info("Validating rewrite decision...")
            
            # Import prompt builder
            from backend.compiler.prompts.critique_prompts import build_rewrite_decision_validation_prompt
            
            # Build validation prompt
            prompt = build_rewrite_decision_validation_prompt(
                user_prompt=user_prompt,
                current_body=current_body,
                current_outline=current_outline,
                current_title=current_title,
                critique_feedback=critique_feedback,
                decision_result=decision_result,
                aggregator_db=aggregator_db
            )
            
            # Generate task ID
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM
            from backend.shared.config import system_config
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=system_config.compiler_validator_max_output_tokens
            )
            
            # Notify task completed
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            # Extract content from response (handles both 'content' and 'reasoning' fields)
            message = response.get("choices", [{}])[0].get("message", {})
            llm_output = message.get("content", "") or message.get("reasoning", "")
            
            # Parse the extracted string
            data = parse_json(llm_output)
            
            if data is None:
                logger.error("Failed to parse rewrite decision validation response")
                return False
            
            # Handle array responses
            if isinstance(data, list):
                logger.warning("Validator returned array instead of object - using first element")
                if not data:
                    return False
                data = data[0]
            
            # Check decision
            decision = data.get("decision", "").lower()
            reasoning = data.get("reasoning", "")
            
            if decision == "accept":
                logger.info(f"Rewrite decision VALIDATED: {reasoning[:200]}...")
                return True
            else:
                logger.info(f"Rewrite decision REJECTED: {reasoning[:200]}...")
                return False
                
        except Exception as e:
            logger.error(f"Error validating rewrite decision: {e}", exc_info=True)
            return False
    
    async def validate_partial_revision_edit(
        self,
        edit_proposal: Dict,
        current_paper: str,
        current_outline: str,
        critique_feedback: str
    ) -> Tuple[bool, str]:
        """
        Validate a single edit proposed during iterative partial revision.
        
        This validates that an edit:
        1. Uses exact string matching correctly
        2. Addresses critique feedback appropriately
        3. Maintains document coherence
        4. Preserves mathematical rigor
        
        Args:
            edit_proposal: Dict with operation, old_string, new_string, reasoning
            current_paper: Current paper state
            current_outline: Paper outline
            critique_feedback: The accepted critique feedback being addressed
            
        Returns:
            Tuple of (is_valid: bool, rejection_reason: str)
        """
        try:
            logger.info("Validating partial revision edit...")
            
            operation = edit_proposal.get("operation", "")
            old_string = edit_proposal.get("old_string", "")
            new_string = edit_proposal.get("new_string", "")
            reasoning = edit_proposal.get("reasoning", "")
            
            # Pre-validation: Check exact string match for non-full_content operations
            if operation in ("replace", "insert_after", "delete"):
                if not old_string:
                    return False, "old_string cannot be empty for this operation"
                
                # Normalize and check
                normalized_paper = normalize_unicode_hyphens(current_paper)
                normalized_old = normalize_unicode_hyphens(old_string)
                
                if normalized_old not in normalized_paper:
                    # Try to find similar text for better error message
                    logger.warning(f"Exact string not found in document: '{old_string[:100]}...'")
                    return False, f"EXACT_STRING_NOT_FOUND: The old_string was not found in the document. Ensure you use text that exists verbatim in CURRENT PAPER."
                
                # Check uniqueness
                count = normalized_paper.count(normalized_old)
                if count > 1:
                    return False, f"STRING_NOT_UNIQUE: The old_string appears {count} times in the document. Include more context to make it unique."
            
            # Import prompt builder for LLM validation
            from backend.compiler.prompts.critique_prompts import build_partial_revision_validation_prompt
            
            # Build validation prompt
            prompt = build_partial_revision_validation_prompt(
                current_paper=current_paper,
                current_outline=current_outline,
                critique_feedback=critique_feedback,
                edit_proposal=edit_proposal
            )
            
            # Generate task ID
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM
            from backend.shared.config import system_config
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=system_config.compiler_validator_max_output_tokens
            )
            
            # Notify task completed
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            # Extract content from response
            message = response.get("choices", [{}])[0].get("message", {})
            llm_output = message.get("content", "") or message.get("reasoning", "")
            
            # Parse the response
            data = parse_json(llm_output)
            
            if data is None:
                logger.error("Failed to parse partial revision validation response")
                return False, "Failed to parse validation response"
            
            # Handle array responses
            if isinstance(data, list):
                logger.warning("Validator returned array - using first element")
                if not data:
                    return False, "Empty validation response"
                data = data[0]
            
            # Check decision
            decision = data.get("decision", "").lower()
            val_reasoning = data.get("reasoning", "No reason provided")
            
            if decision == "accept":
                logger.info(f"Partial revision edit VALIDATED: {val_reasoning[:150]}...")
                return True, ""
            else:
                logger.info(f"Partial revision edit REJECTED: {val_reasoning[:150]}...")
                return False, val_reasoning
                
        except Exception as e:
            logger.error(f"Error validating partial revision edit: {e}", exc_info=True)
            return False, f"Validation error: {str(e)}"