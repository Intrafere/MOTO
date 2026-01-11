"""
API Client Manager - Unified manager for routing API calls to OpenRouter or LM Studio.
Handles fallback on credit exhaustion and boost integration.

Supports three boost modes:
1. Boost Next X Calls - Counter-based, applies to next X API calls
2. Category Boost - Role-based, boosts all calls for specific role categories
3. Per-task Toggle - Task ID based (legacy)
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.openrouter_client import OpenRouterClient, CreditExhaustionError
from backend.shared.boost_manager import boost_manager
from backend.shared.boost_logger import boost_logger
from backend.shared.models import ModelConfig

logger = logging.getLogger(__name__)


class APIClientManager:
    """
    Central manager for routing API calls to OpenRouter or LM Studio.
    Handles fallback on credit exhaustion and boost integration.
    """
    
    def __init__(self):
        self._openrouter_client: Optional[OpenRouterClient] = None
        self._openrouter_api_key: Optional[str] = None
        
        # Track which roles have fallen back to LM Studio
        # Format: {role_id: "openrouter" | "lm_studio"}
        self._role_fallback_state: Dict[str, str] = {}
        
        # Track model configurations per role
        # Format: {role_id: ModelConfig}
        self._role_model_configs: Dict[str, ModelConfig] = {}
        
        # WebSocket broadcaster
        self._broadcast_callback: Optional[Callable] = None
        
        # Model tracking callback for Tier 3
        # Called after each successful API call with the model ID used
        # Signature: async callback(model_id: str)
        self._model_tracking_callback: Optional[Callable] = None
        
        # Autonomous API logger callback
        # Called after each API call (success or failure) with full details
        # Signature: async callback(task_id, role_id, model, provider, prompt, response, duration_ms, success, error, phase)
        self._autonomous_logger_callback: Optional[Callable] = None
        
        # Current autonomous phase (set by autonomous coordinator)
        self._current_autonomous_phase: str = "unknown"
        
        # Lock for thread-safe state updates
        self._state_lock = asyncio.Lock()
    
    def set_broadcast_callback(self, callback: Callable) -> None:
        """Set callback for broadcasting WebSocket events."""
        self._broadcast_callback = callback
    
    async def _broadcast(self, event: str, data: Dict[str, Any] = None) -> None:
        """Broadcast an event through WebSocket."""
        if self._broadcast_callback:
            await self._broadcast_callback(event, data or {})
    
    def set_model_tracking_callback(self, callback: Optional[Callable]) -> None:
        """
        Set callback for model usage tracking during Tier 3 final answer generation.
        
        The callback is called after each successful API call with the model ID used.
        Used to track which models contribute to the final answer and tally API calls.
        
        Args:
            callback: Async function that takes model_id (str) as argument, or None to disable
        """
        self._model_tracking_callback = callback
        if callback:
            logger.info("Model tracking callback set for Tier 3")
        else:
            logger.info("Model tracking callback cleared")
    
    def set_autonomous_logger_callback(self, callback: Optional[Callable]) -> None:
        """
        Set callback for autonomous API logging.
        
        The callback is called after each API call with full details for logging.
        
        Args:
            callback: Async function with signature: 
                      callback(task_id, role_id, model, provider, prompt, response, 
                               duration_ms, success, error, phase)
                      or None to disable
        """
        self._autonomous_logger_callback = callback
        if callback:
            logger.info("Autonomous API logger callback set")
        else:
            logger.info("Autonomous API logger callback cleared")
    
    def set_autonomous_phase(self, phase: str) -> None:
        """
        Set the current autonomous research phase for logging context.
        
        Args:
            phase: Phase identifier ("topic_selection", "brainstorm", "paper_compilation", "tier3")
        """
        self._current_autonomous_phase = phase
    
    async def _track_model_usage(self, model_id: str) -> None:
        """
        Track model usage if tracking callback is set.
        
        Args:
            model_id: The model ID that was used for the API call
        """
        if self._model_tracking_callback:
            try:
                await self._model_tracking_callback(model_id)
            except Exception as e:
                logger.error(f"Error in model tracking callback: {e}")
    
    def set_openrouter_api_key(self, api_key: str) -> None:
        """
        Set OpenRouter API key and initialize client.
        
        Args:
            api_key: OpenRouter API key
        """
        self._openrouter_api_key = api_key
        if api_key:
            self._openrouter_client = OpenRouterClient(api_key)
            logger.info("OpenRouter client initialized")
        else:
            self._openrouter_client = None
            logger.info("OpenRouter client disabled (no API key)")
    
    def configure_role(self, role_id: str, config: ModelConfig) -> None:
        """
        Configure a role with model settings.
        
        Args:
            role_id: Role identifier (e.g., "aggregator_submitter_1", "compiler_validator")
            config: Model configuration (includes provider, model_id, openrouter_model_id, 
                    lm_studio_fallback_id, and optionally openrouter_provider)
        """
        self._role_model_configs[role_id] = config
        
        # Set initial fallback state based on provider
        if config.provider == "openrouter":
            self._role_fallback_state[role_id] = "openrouter"
        else:
            self._role_fallback_state[role_id] = "lm_studio"
        
        # Log configuration with provider details if OpenRouter
        if config.provider == "openrouter":
            or_model = config.openrouter_model_id or config.model_id
            provider_str = f" via {config.openrouter_provider}" if config.openrouter_provider else ""
            fallback_str = f", fallback={config.lm_studio_fallback_id}" if config.lm_studio_fallback_id else ""
            logger.info(f"Configured role '{role_id}': provider=openrouter, model={or_model}{provider_str}{fallback_str}")
        else:
            logger.info(f"Configured role '{role_id}': provider=lm_studio, model={config.model_id}")
    
    def _determine_boost_mode(self, task_id: str) -> Optional[str]:
        """
        Determine which boost mode (if any) applies to this task.
        
        Returns:
            "next_count", "category", "task_id", or None
        """
        if not boost_manager.boost_config or not boost_manager.boost_config.enabled:
            return None
        
        # Check boost_next_count first (counter-based mode)
        if boost_manager.boost_next_count > 0:
            return "next_count"
        
        # Check category boost (role-based mode)
        role_prefix = boost_manager._extract_role_prefix(task_id)
        if role_prefix in boost_manager.boosted_categories:
            return "category"
        
        # Check exact task ID (legacy per-task mode)
        if task_id in boost_manager.boosted_task_ids:
            return "task_id"
        
        return None
    
    async def generate_completion(
        self,
        task_id: str,
        role_id: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion using the appropriate API.
        
        Routing logic:
        1. Check if task should use boost (via should_use_boost) → Use boost OpenRouter model
        2. Check role fallback state:
           - If "openrouter" and not fallen back → Try OpenRouter
           - If "lm_studio" or fallen back → Use LM Studio
        3. On OpenRouter credit exhaustion → Fall back to LM Studio permanently
        
        Args:
            task_id: Task ID to check boost state
            role_id: Role identifier for fallback tracking
            model: Model identifier (LM Studio format)
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional response format
            **kwargs: Additional arguments
            
        Returns:
            API response dict
        """
        # Check if task should use boost (unified check for all boost modes)
        boost_mode = self._determine_boost_mode(task_id)
        
        if boost_mode and boost_manager.boost_config:
            boost_model = boost_manager.boost_config.boost_model_id
            boost_provider = boost_manager.boost_config.boost_provider
            provider_info = f" via {boost_provider}" if boost_provider else " (auto-routing)"
            logger.info(f"Task {task_id} using boost ({boost_mode}): {boost_model}{provider_info}")
            
            # Get prompt preview for logging
            prompt_preview = ""
            if messages:
                last_message = messages[-1].get("content", "")
                prompt_preview = last_message[:500] if last_message else ""
            
            start_time = time.time()
            
            try:
                # Create temporary client with boost API key
                boost_client = OpenRouterClient(boost_manager.boost_config.openrouter_api_key)
                boost_provider = boost_manager.boost_config.boost_provider
                try:
                    result = await boost_client.generate_completion(
                        model=boost_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens or boost_manager.boost_config.boost_max_output_tokens,
                        response_format=response_format,
                        provider=boost_provider
                    )
                    
                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Extract response content for logging
                    response_content = ""
                    tokens_used = None
                    if result.get("choices"):
                        message = result["choices"][0].get("message", {})
                        response_content = message.get("content", "") or message.get("reasoning", "")
                    if result.get("usage"):
                        tokens_used = result["usage"].get("total_tokens")
                    
                    # Log the boost call
                    await boost_logger.log_boost_call(
                        task_id=task_id,
                        role_id=role_id,
                        model=boost_model,
                        prompt_preview=prompt_preview,
                        response_content=response_content,
                        tokens_used=tokens_used,
                        duration_ms=duration_ms,
                        success=True,
                        boost_mode=boost_mode
                    )
                    
                    # Log to autonomous API logger if callback set
                    if self._autonomous_logger_callback:
                        full_prompt = messages[-1].get("content", "") if messages else ""
                        await self._autonomous_logger_callback(
                            task_id=task_id,
                            role_id=role_id,
                            model=boost_model,
                            provider="openrouter",
                            prompt=full_prompt,
                            response=response_content,
                            tokens_used=tokens_used,
                            duration_ms=duration_ms,
                            success=True,
                            error=None,
                            phase=self._current_autonomous_phase
                        )
                    
                    # Track model usage for Tier 3
                    await self._track_model_usage(boost_model)
                    
                    # Consume boost count if using next_count mode
                    if boost_mode == "next_count":
                        await boost_manager.consume_boost_count()
                    
                    return result
                finally:
                    await boost_client.close()
                    
            except CreditExhaustionError as e:
                # Log the failed boost call
                duration_ms = (time.time() - start_time) * 1000
                await boost_logger.log_boost_call(
                    task_id=task_id,
                    role_id=role_id,
                    model=boost_model,
                    prompt_preview=prompt_preview,
                    response_content="",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    boost_mode=boost_mode
                )
                
                # Log to autonomous API logger if callback set
                if self._autonomous_logger_callback:
                    full_prompt = messages[-1].get("content", "") if messages else ""
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=boost_model,
                        provider="openrouter",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        phase=self._current_autonomous_phase
                    )
                
                # Boost credits exhausted - fall back to primary for this task
                logger.warning(f"Boost credits exhausted for task {task_id}, using primary model")
                await self._broadcast("boost_credits_exhausted", {
                    "task_id": task_id,
                    "message": str(e)
                })
                # Continue to primary model routing below
                
            except Exception as e:
                # Log the failed boost call
                duration_ms = (time.time() - start_time) * 1000
                await boost_logger.log_boost_call(
                    task_id=task_id,
                    role_id=role_id,
                    model=boost_model,
                    prompt_preview=prompt_preview,
                    response_content="",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    boost_mode=boost_mode
                )
                
                # Log to autonomous API logger if callback set
                if self._autonomous_logger_callback:
                    full_prompt = messages[-1].get("content", "") if messages else ""
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=boost_model,
                        provider="openrouter",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        phase=self._current_autonomous_phase
                    )
                
                logger.error(f"Boost API error for task {task_id}: {e}, using primary model")
                # Fall through to primary model
        
        # Check role fallback state
        async with self._state_lock:
            fallback_state = self._role_fallback_state.get(role_id, "lm_studio")
            role_config = self._role_model_configs.get(role_id)
        
        # If OpenRouter configured and not fallen back, try OpenRouter
        if fallback_state == "openrouter" and self._openrouter_client and role_config:
            openrouter_model = role_config.openrouter_model_id or role_config.model_id
            openrouter_provider = role_config.openrouter_provider
            
            provider_info = f" via {openrouter_provider}" if openrouter_provider else ""
            
            start_time = time.time()
            
            try:
                logger.debug(f"Role {role_id} using OpenRouter: {openrouter_model}{provider_info}")
                result = await self._openrouter_client.generate_completion(
                    model=openrouter_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens or role_config.max_output_tokens,
                    response_format=response_format,
                    provider=openrouter_provider  # Pass specific provider if configured
                )
                
                # Calculate duration and extract response
                duration_ms = (time.time() - start_time) * 1000
                response_content = ""
                tokens_used = None
                if result.get("choices"):
                    message = result["choices"][0].get("message", {})
                    response_content = message.get("content", "") or message.get("reasoning", "")
                if result.get("usage"):
                    tokens_used = result["usage"].get("total_tokens")
                
                # Log to autonomous API logger if callback set
                if self._autonomous_logger_callback:
                    full_prompt = messages[-1].get("content", "") if messages else ""
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=openrouter_model,
                        provider="openrouter",
                        prompt=full_prompt,
                        response=response_content,
                        tokens_used=tokens_used,
                        duration_ms=duration_ms,
                        success=True,
                        error=None,
                        phase=self._current_autonomous_phase
                    )
                
                # Track model usage for Tier 3
                await self._track_model_usage(openrouter_model)
                
                return result
            
            except CreditExhaustionError as e:
                # PERMANENT FALLBACK - OpenRouter credits exhausted for this role
                duration_ms = (time.time() - start_time) * 1000
                
                # Log to autonomous API logger if callback set
                if self._autonomous_logger_callback:
                    full_prompt = messages[-1].get("content", "") if messages else ""
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=openrouter_model,
                        provider="openrouter",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=f"Credit Exhaustion: {str(e)}",
                        phase=self._current_autonomous_phase
                    )
                
                # CHECK: Is fallback configured?
                if not role_config.lm_studio_fallback_id:
                    # NO FALLBACK - raise clear error
                    error_msg = (
                        f"OpenRouter credits exhausted for role '{role_id}' "
                        f"and no LM Studio fallback configured. "
                        f"Please add credits to OpenRouter or configure an LM Studio "
                        f"fallback model in settings."
                    )
                    logger.error(error_msg)
                    await self._broadcast("openrouter_fallback_failed", {
                        "role_id": role_id,
                        "reason": "no_fallback_configured",
                        "message": error_msg
                    })
                    raise RuntimeError(error_msg)
                
                # Fallback IS configured - use it
                async with self._state_lock:
                    self._role_fallback_state[role_id] = "lm_studio"
                
                fallback_model = role_config.lm_studio_fallback_id
                
                logger.error(
                    f"OpenRouter credits exhausted for role '{role_id}'. "
                    f"Permanently falling back to LM Studio model: {fallback_model}"
                )
                
                await self._broadcast("openrouter_fallback", {
                    "role_id": role_id,
                    "reason": "credit_exhaustion",
                    "message": str(e),
                    "fallback_model": fallback_model
                })
                
                # Fall through to LM Studio
                model = fallback_model
            
            except Exception as e:
                # Other OpenRouter error - fall back for this call only (don't mark as permanent)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log to autonomous API logger if callback set
                if self._autonomous_logger_callback:
                    full_prompt = messages[-1].get("content", "") if messages else ""
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=openrouter_model,
                        provider="openrouter",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        phase=self._current_autonomous_phase
                    )
                
                # For non-credit errors, only fall back if fallback is configured
                if role_config.lm_studio_fallback_id:
                    logger.error(
                        f"OpenRouter error for role '{role_id}': {e}, "
                        f"falling back to LM Studio model: {role_config.lm_studio_fallback_id}"
                    )
                    model = role_config.lm_studio_fallback_id
                    # Fall through to LM Studio
                else:
                    # No fallback configured - re-raise the error
                    logger.error(
                        f"OpenRouter error for role '{role_id}': {e}, "
                        f"and no LM Studio fallback configured"
                    )
                    raise
        
        # Use LM Studio (either configured as primary or fallen back)
        logger.debug(f"Role {role_id} using LM Studio: {model}")
        start_time = time.time()
        
        try:
            result = await lm_studio_client.generate_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **kwargs
            )
            
            # Calculate duration and extract response
            duration_ms = (time.time() - start_time) * 1000
            response_content = ""
            tokens_used = None
            if result.get("choices"):
                message = result["choices"][0].get("message", {})
                response_content = message.get("content", "") or message.get("reasoning", "")
            if result.get("usage"):
                tokens_used = result["usage"].get("total_tokens")
            
            # Log to autonomous API logger if callback set
            if self._autonomous_logger_callback:
                full_prompt = messages[-1].get("content", "") if messages else ""
                await self._autonomous_logger_callback(
                    task_id=task_id,
                    role_id=role_id,
                    model=model,
                    provider="lm_studio",
                    prompt=full_prompt,
                    response=response_content,
                    tokens_used=tokens_used,
                    duration_ms=duration_ms,
                    success=True,
                    error=None,
                    phase=self._current_autonomous_phase
                )
            
            # Track model usage for Tier 3
            await self._track_model_usage(model)
            
            return result
            
        except Exception as e:
            # Log LM Studio error to autonomous logger if callback set
            duration_ms = (time.time() - start_time) * 1000
            if self._autonomous_logger_callback:
                full_prompt = messages[-1].get("content", "") if messages else ""
                await self._autonomous_logger_callback(
                    task_id=task_id,
                    role_id=role_id,
                    model=model,
                    provider="lm_studio",
                    prompt=full_prompt,
                    response="",
                    tokens_used=None,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    phase=self._current_autonomous_phase
                )
            # Re-raise the exception
            raise
    
    def get_fallback_state(self, role_id: str) -> str:
        """
        Get current fallback state for a role.
        
        Args:
            role_id: Role identifier
            
        Returns:
            "openrouter" or "lm_studio"
        """
        return self._role_fallback_state.get(role_id, "lm_studio")
    
    def get_all_fallback_states(self) -> Dict[str, str]:
        """
        Get fallback states for all configured roles.
        
        Returns:
            Dict mapping role_id to fallback state
        """
        return self._role_fallback_state.copy()
    
    async def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        Get embeddings, routing to LM Studio first, then OpenRouter fallback.
        
        This enables the system to work without LM Studio if OpenRouter is configured.
        LM Studio is tried first (local, free), then falls back to OpenRouter.
        
        Args:
            texts: Texts to embed
            model: Optional model override
        
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If both LM Studio and OpenRouter are unavailable
        """
        if not texts:
            return []
        
        # Try LM Studio first (local, free)
        try:
            return await lm_studio_client.get_embeddings(texts, model)
        except Exception as lm_error:
            logger.warning(f"LM Studio embeddings unavailable: {lm_error}")
            
            # Fall back to OpenRouter if configured
            if self._openrouter_client:
                logger.info("Falling back to OpenRouter for embeddings")
                try:
                    return await self._openrouter_client.get_embeddings(texts, model)
                except Exception as or_error:
                    logger.error(f"OpenRouter embeddings also failed: {or_error}")
                    raise RuntimeError(
                        f"Embeddings unavailable: LM Studio error ({lm_error}), "
                        f"OpenRouter error ({or_error})"
                    )
            else:
                raise RuntimeError(
                    "Embeddings unavailable: LM Studio is down and OpenRouter is not configured. "
                    "Please start LM Studio or configure OpenRouter API key."
                )
    
    async def close(self):
        """Close all API clients."""
        if self._openrouter_client:
            await self._openrouter_client.close()
        # lm_studio_client is global singleton, don't close it here


# Global singleton instance
api_client_manager = APIClientManager()

