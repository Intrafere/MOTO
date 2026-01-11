"""
Paper Critique Memory Module.

Handles persistence of paper critiques from the validator model.
Supports three paper types: autonomous_paper, final_answer, compiler_paper.
Each paper can have up to 10 critiques stored (oldest removed when exceeded).

DUAL-PATH ARCHITECTURE
======================
The autonomous research system uses two storage modes:

1. LEGACY PATHS (backward compatibility):
   - Papers: backend/data/auto_papers/
   - Final answers: backend/data/auto_final_answer/
   - Used when existing legacy data is detected

2. SESSION-BASED PATHS (preferred for new sessions):
   - Papers: backend/data/auto_sessions/{session_id}/papers/
   - Final answers: backend/data/auto_sessions/{session_id}/final_answer/
   - Created for new research sessions

HOW TO USE base_path PARAMETER:
- For session-based papers: Pass the paper's directory from paper_library.get_paper_path()
- For final answers: Pass the final answer's base directory
- If base_path is None, falls back to legacy paths (for backward compatibility)

The compiler paper type always uses a single global file and ignores base_path.
"""

import json
import os
import logging
from typing import List, Optional, Literal
from datetime import datetime
import uuid

from backend.shared.models import PaperCritique

logger = logging.getLogger(__name__)

# Maximum number of critiques to store per paper
MAX_CRITIQUES_PER_PAPER = 10

# Paper type definitions
PaperType = Literal["autonomous_paper", "final_answer", "compiler_paper"]


def _get_critiques_file_path(
    paper_type: PaperType,
    paper_id: Optional[str] = None,
    base_path: Optional[str] = None
) -> str:
    """
    Get the file path for storing critiques based on paper type.
    
    Args:
        paper_type: Type of paper ("autonomous_paper", "final_answer", "compiler_paper")
        paper_id: Required for autonomous_paper type (used in filename)
        base_path: Optional override path. If provided, critiques are stored here
                   instead of legacy paths. This enables session-aware storage.
        
    Returns:
        Path to the critiques JSON file
    """
    # If base_path is provided, use it for session-aware storage
    if base_path:
        os.makedirs(base_path, exist_ok=True)
        
        if paper_type == "autonomous_paper":
            if not paper_id:
                raise ValueError("paper_id is required for autonomous_paper type")
            return os.path.join(base_path, f"paper_{paper_id}_critiques.json")
        
        elif paper_type == "final_answer":
            # Final answer critiques stored in the final answer directory
            return os.path.join(base_path, "final_answer_critiques.json")
        
        elif paper_type == "compiler_paper":
            # Compiler always uses global path (ignore base_path)
            pass  # Fall through to legacy handling
        
        else:
            raise ValueError(f"Unknown paper_type: {paper_type}")
    
    # Legacy paths (fallback when base_path not provided)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    if paper_type == "autonomous_paper":
        if not paper_id:
            raise ValueError("paper_id is required for autonomous_paper type")
        papers_dir = os.path.join(data_dir, "auto_papers")
        os.makedirs(papers_dir, exist_ok=True)
        return os.path.join(papers_dir, f"paper_{paper_id}_critiques.json")
    
    elif paper_type == "final_answer":
        final_answer_dir = os.path.join(data_dir, "auto_final_answer")
        os.makedirs(final_answer_dir, exist_ok=True)
        return os.path.join(final_answer_dir, "final_answer_critiques.json")
    
    elif paper_type == "compiler_paper":
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "compiler_paper_critiques.json")
    
    else:
        raise ValueError(f"Unknown paper_type: {paper_type}")


async def save_critique(
    paper_type: PaperType,
    critique: PaperCritique,
    paper_id: Optional[str] = None,
    base_path: Optional[str] = None
) -> PaperCritique:
    """
    Save a critique to the paper's critique history.
    
    Maintains a maximum of MAX_CRITIQUES_PER_PAPER critiques per paper.
    Oldest critiques are removed when the limit is exceeded.
    
    Args:
        paper_type: Type of paper
        critique: The PaperCritique to save
        paper_id: Required for autonomous_paper type
        base_path: Optional override path for session-aware storage.
                   If provided, critiques are stored in this directory.
                   If None, falls back to legacy paths.
        
    Returns:
        The saved PaperCritique (with generated critique_id if not set)
    """
    file_path = _get_critiques_file_path(paper_type, paper_id, base_path)
    
    # Ensure critique has an ID
    if not critique.critique_id:
        critique.critique_id = str(uuid.uuid4())[:8]
    
    # Load existing critiques
    critiques = await get_critiques(paper_type, paper_id, base_path)
    
    # Add new critique at the beginning (newest first)
    critiques.insert(0, critique)
    
    # Enforce max limit (remove oldest)
    while len(critiques) > MAX_CRITIQUES_PER_PAPER:
        removed = critiques.pop()
        logger.info(f"Removed oldest critique {removed.critique_id} to maintain limit of {MAX_CRITIQUES_PER_PAPER}")
    
    # Save to file
    critiques_data = [c.model_dump() for c in critiques]
    
    # Convert datetime objects to ISO format strings for JSON serialization
    for c in critiques_data:
        if isinstance(c.get("date"), datetime):
            c["date"] = c["date"].isoformat()
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(critiques_data, f, indent=2, default=str)
        logger.info(f"Saved critique {critique.critique_id} for {paper_type}" + 
                   (f" paper_id={paper_id}" if paper_id else "") +
                   (f" at {file_path}" if base_path else ""))
    except Exception as e:
        logger.error(f"Failed to save critique: {e}")
        raise
    
    return critique


async def get_critiques(
    paper_type: PaperType,
    paper_id: Optional[str] = None,
    base_path: Optional[str] = None
) -> List[PaperCritique]:
    """
    Get all critiques for a paper.
    
    Args:
        paper_type: Type of paper
        paper_id: Required for autonomous_paper type
        base_path: Optional override path for session-aware storage.
                   If provided, looks for critiques in this directory.
                   If None, falls back to legacy paths.
        
    Returns:
        List of PaperCritique objects (newest first)
    """
    file_path = _get_critiques_file_path(paper_type, paper_id, base_path)
    
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            critiques_data = json.load(f)
        
        critiques = []
        for c in critiques_data:
            # Convert ISO format strings back to datetime
            if isinstance(c.get("date"), str):
                try:
                    c["date"] = datetime.fromisoformat(c["date"])
                except ValueError:
                    c["date"] = datetime.now()
            critiques.append(PaperCritique(**c))
        
        return critiques
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse critiques file {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load critiques from {file_path}: {e}")
        return []


async def clear_critiques(
    paper_type: PaperType,
    paper_id: Optional[str] = None,
    base_path: Optional[str] = None
) -> bool:
    """
    Delete all critiques for a paper.
    
    Args:
        paper_type: Type of paper
        paper_id: Required for autonomous_paper type
        base_path: Optional override path for session-aware storage.
                   If provided, deletes critiques from this directory.
                   If None, falls back to legacy paths.
        
    Returns:
        True if file was deleted, False if it didn't exist
    """
    file_path = _get_critiques_file_path(paper_type, paper_id, base_path)
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Cleared critiques for {paper_type}" + 
                       (f" paper_id={paper_id}" if paper_id else "") +
                       (f" at {file_path}" if base_path else ""))
            return True
        except Exception as e:
            logger.error(f"Failed to delete critiques file {file_path}: {e}")
            raise
    
    return False


async def get_critique_by_id(
    paper_type: PaperType,
    critique_id: str,
    paper_id: Optional[str] = None,
    base_path: Optional[str] = None
) -> Optional[PaperCritique]:
    """
    Get a specific critique by its ID.
    
    Args:
        paper_type: Type of paper
        critique_id: The critique ID to find
        paper_id: Required for autonomous_paper type
        base_path: Optional override path for session-aware storage
        
    Returns:
        The PaperCritique if found, None otherwise
    """
    critiques = await get_critiques(paper_type, paper_id, base_path)
    
    for critique in critiques:
        if critique.critique_id == critique_id:
            return critique
    
    return None


async def get_latest_critique(
    paper_type: PaperType,
    paper_id: Optional[str] = None,
    base_path: Optional[str] = None
) -> Optional[PaperCritique]:
    """
    Get the most recent critique for a paper.
    
    Args:
        paper_type: Type of paper
        paper_id: Required for autonomous_paper type
        base_path: Optional override path for session-aware storage
        
    Returns:
        The most recent PaperCritique if any exist, None otherwise
    """
    critiques = await get_critiques(paper_type, paper_id, base_path)
    
    if critiques:
        return critiques[0]  # Already sorted newest first
    
    return None
