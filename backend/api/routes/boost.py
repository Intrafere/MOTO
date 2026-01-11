"""
API routes for boost management.

Supports three boost modes:
1. Boost Next X Calls - Counter-based (/api/boost/set-next-count)
2. Category Boost - Role-based (/api/boost/toggle-category/{category})
3. Per-task Toggle - Task ID based (/api/boost/toggle-task/{task_id})

Plus boost logging endpoints for viewing API call history.
"""
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from backend.shared.models import BoostConfig
from backend.shared.boost_manager import boost_manager
from backend.shared.boost_logger import boost_logger
from backend.shared.openrouter_client import OpenRouterClient

router = APIRouter()
logger = logging.getLogger(__name__)


class BoostNextCountRequest(BaseModel):
    """Request body for setting boost next count."""
    count: int


@router.post("/api/boost/enable")
async def enable_boost(config: BoostConfig) -> Dict[str, Any]:
    """
    Enable API boost with OpenRouter.
    
    Args:
        config: Boost configuration with API key and model
        
    Returns:
        Status and boost configuration
    """
    try:
        # Validate API key by testing connection
        if not config.openrouter_api_key:
            raise HTTPException(status_code=400, detail="OpenRouter API key is required")
        
        # Test connection
        client = OpenRouterClient(config.openrouter_api_key)
        models = await client.list_models()
        
        if not models:
            raise HTTPException(
                status_code=400,
                detail="Failed to connect to OpenRouter. Please check your API key."
            )
        
        await client.close()
        
        # Enable boost
        await boost_manager.set_boost_config(config)
        
        provider_info = f", provider={config.boost_provider}" if config.boost_provider else " (auto-routing)"
        logger.info(f"Boost enabled: model={config.boost_model_id}{provider_info}")
        
        return {
            "success": True,
            "message": "Boost enabled successfully",
            "config": {
                "model_id": config.boost_model_id,
                "provider": config.boost_provider,
                "context_window": config.boost_context_window,
                "max_output_tokens": config.boost_max_output_tokens
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable boost: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable boost: {str(e)}")


@router.post("/api/boost/update-model")
async def update_boost_model(config: BoostConfig) -> Dict[str, Any]:
    """
    Update boost model/API key WITHOUT clearing boost state.
    
    This allows seamless model switching while preserving:
    - boost_next_count
    - boosted_categories
    - boosted_task_ids
    
    Args:
        config: New boost configuration with API key and model
        
    Returns:
        Status and updated configuration
    """
    try:
        # Validate that boost is currently enabled
        if not boost_manager.boost_config or not boost_manager.boost_config.enabled:
            raise HTTPException(
                status_code=400, 
                detail="Boost must be enabled first. Use /api/boost/enable to enable boost."
            )
        
        # Validate API key by testing connection
        if not config.openrouter_api_key:
            raise HTTPException(status_code=400, detail="OpenRouter API key is required")
        
        # Test connection with new model
        client = OpenRouterClient(config.openrouter_api_key)
        models = await client.list_models()
        
        if not models:
            raise HTTPException(
                status_code=400,
                detail="Failed to connect to OpenRouter. Please check your API key."
            )
        
        await client.close()
        
        # Store current boost state before update
        old_boost_next_count = boost_manager.boost_next_count
        old_boosted_categories = boost_manager.boosted_categories.copy()
        old_boosted_task_ids = boost_manager.boosted_task_ids.copy()
        
        # Update config (preserves boost state automatically)
        await boost_manager.set_boost_config(config)
        
        # Log the change
        provider_info = f", provider={config.boost_provider}" if config.boost_provider else " (auto-routing)"
        logger.info(
            f"Boost model updated: {config.boost_model_id}{provider_info}\n"
            f"  Preserved state: boost_next_count={old_boost_next_count}, "
            f"boosted_categories={len(old_boosted_categories)}, "
            f"boosted_tasks={len(old_boosted_task_ids)}"
        )
        
        return {
            "success": True,
            "message": "Boost model updated successfully (state preserved)",
            "config": {
                "model_id": config.boost_model_id,
                "provider": config.boost_provider,
                "context_window": config.boost_context_window,
                "max_output_tokens": config.boost_max_output_tokens
            },
            "preserved_state": {
                "boost_next_count": boost_manager.boost_next_count,
                "boosted_categories": list(boost_manager.boosted_categories),
                "boosted_task_count": len(boost_manager.boosted_task_ids)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update boost model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


@router.post("/api/boost/disable")
async def disable_boost() -> Dict[str, Any]:
    """
    Disable API boost.
    
    Returns:
        Status message
    """
    try:
        await boost_manager.clear_boost()
        logger.info("Boost disabled")
        
        return {
            "success": True,
            "message": "Boost disabled successfully"
        }
    except Exception as e:
        logger.error(f"Failed to disable boost: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disable boost: {str(e)}")


@router.get("/api/boost/status")
async def get_boost_status() -> Dict[str, Any]:
    """
    Get current boost status.
    
    Returns:
        Boost configuration and active tasks
    """
    try:
        status = boost_manager.get_boost_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"Failed to get boost status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get boost status: {str(e)}")


@router.post("/api/boost/toggle-task/{task_id}")
async def toggle_task_boost(task_id: str) -> Dict[str, Any]:
    """
    Toggle boost for a specific task.
    
    Args:
        task_id: Task ID to toggle
        
    Returns:
        New boost state for the task
    """
    try:
        boosted = await boost_manager.toggle_task_boost(task_id)
        
        return {
            "success": True,
            "task_id": task_id,
            "boosted": boosted
        }
    except Exception as e:
        logger.error(f"Failed to toggle task boost: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to toggle task boost: {str(e)}")


@router.get("/api/boost/openrouter-models")
async def get_openrouter_models(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Fetch available OpenRouter models.
    
    Args:
        authorization: OpenRouter API key via Authorization header (Bearer token)
        
    Returns:
        List of available models
    """
    try:
        # Extract API key from Authorization header
        api_key = authorization.replace("Bearer ", "") if authorization and authorization.startswith("Bearer ") else authorization
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required in Authorization header")
        
        client = OpenRouterClient(api_key)
        models = await client.list_models()
        await client.close()
        
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        logger.error(f"Failed to fetch OpenRouter models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")


@router.get("/api/boost/model-providers")
async def get_model_providers(model_id: str, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Fetch available providers for a specific OpenRouter model.
    
    Args:
        model_id: The model ID to get providers for (query parameter)
        authorization: OpenRouter API key via Authorization header (Bearer token)
        
    Returns:
        List of available providers for the model
    """
    try:
        # Extract API key from Authorization header
        api_key = authorization.replace("Bearer ", "") if authorization and authorization.startswith("Bearer ") else authorization
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required in Authorization header")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
        
        client = OpenRouterClient(api_key)
        providers = await client.get_model_providers(model_id)
        await client.close()
        
        return {
            "success": True,
            "model_id": model_id,
            "providers": providers
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch providers for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch providers: {str(e)}")


# ============================================================
# NEW: Boost Next X Calls (Counter-based mode)
# ============================================================

@router.post("/api/boost/set-next-count")
async def set_boost_next_count(request: BoostNextCountRequest) -> Dict[str, Any]:
    """
    Set the number of next API calls to boost.
    
    This mode boosts the next X API calls regardless of task ID or category.
    The counter decrements after each boosted call.
    
    Args:
        request: Request with count field
        
    Returns:
        Success status and new count
    """
    try:
        if request.count < 0:
            raise HTTPException(status_code=400, detail="Count must be non-negative")
        
        if not boost_manager.boost_config or not boost_manager.boost_config.enabled:
            raise HTTPException(status_code=400, detail="Boost must be enabled first")
        
        await boost_manager.set_boost_next_count(request.count)
        
        logger.info(f"Boost next count set to {request.count}")
        
        return {
            "success": True,
            "message": f"Will boost the next {request.count} API calls",
            "count": request.count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set boost next count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set count: {str(e)}")


# ============================================================
# NEW: Category Boost (Role-based mode)
# ============================================================

@router.post("/api/boost/toggle-category/{category}")
async def toggle_category_boost(category: str) -> Dict[str, Any]:
    """
    Toggle boost for an entire category (role prefix).
    
    When a category is boosted, ALL API calls for that role will use boost.
    
    Categories:
    - Aggregator: agg_sub1, agg_sub2, ..., agg_sub10, agg_val
    - Compiler: comp_hc, comp_hp, comp_val
    - Autonomous: auto_ts, auto_tv, auto_cr, auto_rs, auto_pt, auto_prc
    
    Args:
        category: Category prefix to toggle
        
    Returns:
        New boost state for the category
    """
    try:
        if not boost_manager.boost_config or not boost_manager.boost_config.enabled:
            raise HTTPException(status_code=400, detail="Boost must be enabled first")
        
        boosted = await boost_manager.toggle_category_boost(category)
        
        return {
            "success": True,
            "category": category,
            "boosted": boosted,
            "all_boosted_categories": list(boost_manager.boosted_categories)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle category boost: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to toggle category: {str(e)}")


@router.get("/api/boost/categories")
async def get_boost_categories(mode: Optional[str] = "all") -> Dict[str, Any]:
    """
    Get available boost categories for the current workflow mode.
    
    Args:
        mode: "aggregator", "compiler", "autonomous", or "all" (default)
        
    Returns:
        List of available categories with their current boost state
    """
    try:
        categories = boost_manager.get_available_categories(mode)
        
        # Add boosted state to each category
        for cat in categories:
            cat["boosted"] = cat["id"] in boost_manager.boosted_categories
        
        return {
            "success": True,
            "categories": categories,
            "boosted_categories": list(boost_manager.boosted_categories)
        }
    except Exception as e:
        logger.error(f"Failed to get boost categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


# ============================================================
# NEW: Boost Logs
# ============================================================

@router.get("/api/boost/logs")
async def get_boost_logs(limit: int = 100) -> Dict[str, Any]:
    """
    Get recent boost API call logs.
    
    Args:
        limit: Maximum number of log entries to return (default 100)
        
    Returns:
        List of log entries (most recent first)
    """
    try:
        logs = await boost_logger.get_logs(limit)
        stats = await boost_logger.get_stats()
        
        return {
            "success": True,
            "logs": logs,
            "stats": stats,
            "total": len(logs)
        }
    except Exception as e:
        logger.error(f"Failed to get boost logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.get("/api/boost/logs/{index}")
async def get_boost_log_entry(index: int) -> Dict[str, Any]:
    """
    Get a specific log entry with full response content.
    
    Args:
        index: Index of the log entry (0 = most recent)
        
    Returns:
        Full log entry including complete response
    """
    try:
        entry = await boost_logger.get_log_entry(index)
        
        if not entry:
            raise HTTPException(status_code=404, detail="Log entry not found")
        
        return {
            "success": True,
            "entry": entry
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get boost log entry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entry: {str(e)}")


@router.post("/api/boost/clear-logs")
async def clear_boost_logs() -> Dict[str, Any]:
    """
    Clear all boost API logs.
    
    Returns:
        Success status
    """
    try:
        await boost_logger.clear_logs()
        
        logger.info("Boost logs cleared")
        
        return {
            "success": True,
            "message": "Boost logs cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear boost logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")

