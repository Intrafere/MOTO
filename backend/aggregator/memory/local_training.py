"""
Per-submitter rejection logs.
Tracks last 5 rejections to help submitters learn from mistakes.
"""
import aiofiles
from pathlib import Path
from typing import List
import asyncio
import logging

from backend.shared.config import system_config, rag_config
from backend.shared.utils import truncate_with_ellipsis

logger = logging.getLogger(__name__)


class LocalTrainingMemory:
    """
    Per-submitter rejection log.
    Maintains rolling window of last 5 rejections.
    """
    
    def __init__(self, submitter_id: int):
        self.submitter_id = submitter_id
        self.file_path = Path(
            f"{system_config.data_dir}/"
            f"Summary_Of_Last_5_Validator_Rejections_For_Submitter_{submitter_id}.txt"
        )
        self.rejections: List[dict] = []
        self.max_rejections = rag_config.max_local_rejections
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize local training memory."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.file_path.exists():
            # Load existing rejections
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                if content.strip():
                    # Parse rejection entries
                    entries = content.split('\n---\n')
                    for entry in entries:
                        if entry.strip():
                            parts = entry.split('\n[SUBMISSION PREVIEW]\n')
                            if len(parts) == 2:
                                self.rejections.append({
                                    'validator_summary': parts[0].replace('[VALIDATOR SUMMARY]\n', '').strip(),
                                    'submission_preview': parts[1].strip()
                                })
            logger.info(f"Loaded {len(self.rejections)} rejections for submitter {self.submitter_id}")
        else:
            # Create empty file
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write("")
            logger.info(f"Created new rejection log for submitter {self.submitter_id}")
    
    async def add_rejection(
        self,
        validator_summary: str,
        submission_content: str
    ) -> None:
        """
        Add a rejection to the log.
        
        Args:
            validator_summary: Validator's reasoning (max 750 chars)
            submission_content: Original submission (first 750 chars)
        """
        async with self._lock:
            # Truncate to limits
            summary = truncate_with_ellipsis(validator_summary, 750)
            preview = truncate_with_ellipsis(submission_content, 750)
            
            # Add rejection
            self.rejections.append({
                'validator_summary': summary,
                'submission_preview': preview
            })
            
            # Keep only last N rejections
            if len(self.rejections) > self.max_rejections:
                self.rejections.pop(0)
            
            # Save to file
            await self._save()
    
    async def reset(self) -> None:
        """Reset (clear) all rejections."""
        async with self._lock:
            self.rejections = []
            await self._save()
            logger.info(f"Reset rejection log for submitter {self.submitter_id}")
    
    async def clear(self) -> None:
        """Alias for reset() - clear all rejections."""
        await self.reset()
    
    async def get_all_content(self) -> str:
        """Get all rejection content as a single string."""
        async with self._lock:
            if not self.rejections:
                return "No rejections yet."
            
            entries = []
            for idx, rejection in enumerate(self.rejections, start=1):
                entry = (
                    f"[REJECTION {idx}]\n"
                    f"[VALIDATOR SUMMARY]\n"
                    f"{rejection['validator_summary']}\n\n"
                    f"[SUBMISSION PREVIEW]\n"
                    f"{rejection['submission_preview']}"
                )
                entries.append(entry)
            
            return '\n\n---\n\n'.join(entries)
    
    async def get_count(self) -> int:
        """Get number of rejections."""
        async with self._lock:
            return len(self.rejections)
    
    async def _save(self) -> None:
        """Save rejections to file."""
        if not self.rejections:
            content = ""
        else:
            entries = []
            for rejection in self.rejections:
                entry = (
                    f"[VALIDATOR SUMMARY]\n"
                    f"{rejection['validator_summary']}\n"
                    f"[SUBMISSION PREVIEW]\n"
                    f"{rejection['submission_preview']}"
                )
                entries.append(entry)
            content = '\n---\n'.join(entries)
        
        async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        logger.debug(f"Saved {len(self.rejections)} rejections for submitter {self.submitter_id}")


# Local training memories are created per-submitter by coordinator

