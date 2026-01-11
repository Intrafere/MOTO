"""
Paper memory manager for compiler.
Handles paper file I/O, real-time updates, and word count tracking.
"""
import aiofiles
import asyncio
from typing import Optional, Callable, List, Dict
from pathlib import Path
import logging

from backend.shared.config import system_config

logger = logging.getLogger(__name__)

# Hardcoded paper anchor - serves as non-chronological stop token
PAPER_ANCHOR = "[HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]"

# Section placeholders - replaced when validated content is written
# These make it crystal clear to the AI where sections go and that they don't exist yet
ABSTRACT_PLACEHOLDER = "[HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION - TO BE WRITTEN AFTER THE INTRODUCTION IS COMPLETE]"
INTRO_PLACEHOLDER = "[HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION - TO BE WRITTEN AFTER THE CONCLUSION SECTION IS COMPLETE]"
CONCLUSION_PLACEHOLDER = "[HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION - TO BE WRITTEN AFTER THE BODY SECTION IS COMPLETE]"


class PaperMemory:
    """
    Manages the current paper state.
    - File I/O to compiler_paper.txt
    - Real-time updates for GUI
    - Word count tracking
    - Thread-safe operations
    - Version tracking
    """
    
    def __init__(self):
        self.file_path = Path(system_config.compiler_paper_file)
        self.version = 0
        self.rechunk_callback: Optional[Callable] = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self.previous_versions = []  # Store previous body versions for UI display
    
    async def initialize(self) -> None:
        """Initialize paper memory."""
        async with self._lock:
            if self._initialized:
                return
            
            # Create file if doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists():
                async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                    await f.write("")
            
            self._initialized = True
            logger.info("Paper memory initialized")
    
    async def _get_paper_unlocked(self) -> str:
        """Get current paper content (internal, assumes lock already held)."""
        if not self.file_path.exists():
            return ""
        
        async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        return content
    
    async def get_paper(self) -> str:
        """Get current paper content."""
        async with self._lock:
            return await self._get_paper_unlocked()
    
    def _strip_duplicate_anchors(self, content: str) -> str:
        """
        Remove all anchor occurrences from content.
        
        Args:
            content: Content to clean
        
        Returns:
            Content with anchors removed
        """
        return content.replace(PAPER_ANCHOR, "").strip()
    
    def _ensure_anchor(self, content: str) -> str:
        """
        Ensure paper ends with anchor marker.
        
        Args:
            content: Paper content
        
        Returns:
            Content with anchor appended at end
        """
        if not content.strip():
            return ""
        
        # Add anchor if not already at end
        content_stripped = content.rstrip()
        if not content_stripped.endswith(PAPER_ANCHOR):
            return content_stripped + "\n\n" + PAPER_ANCHOR
        return content_stripped
    
    async def _update_paper_unlocked(self, new_paper: str) -> str:
        """
        Update paper file (internal, assumes lock already held).
        Does NOT trigger re-chunking callback - caller must handle that.
        
        Args:
            new_paper: New paper content
            
        Returns:
            The final content that was written (for rechunk callback)
        """
        # Strip any duplicate anchors from input
        cleaned = self._strip_duplicate_anchors(new_paper)
        
        # Ensure single anchor at end
        final_content = self._ensure_anchor(cleaned)
        
        # Write to file
        async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
            await f.write(final_content)
        
        # Increment version
        self.version += 1
        
        logger.info(f"Paper updated (version {self.version}, {self.get_word_count_sync(final_content)} words)")
        
        return final_content
    
    async def update_paper(self, new_paper: str) -> None:
        """
        Update paper and trigger re-chunking.
        Automatically strips duplicate anchors and ensures single anchor at end.
        
        Args:
            new_paper: New paper content
        """
        async with self._lock:
            final_content = await self._update_paper_unlocked(new_paper)
        
        # Trigger re-chunking callback OUTSIDE the lock to avoid deadlock
        if self.rechunk_callback:
            try:
                await self.rechunk_callback(final_content)
            except Exception as e:
                logger.error(f"Re-chunking callback failed: {e}")
    
    async def get_word_count(self) -> int:
        """Get current paper word count."""
        content = await self.get_paper()
        return self.get_word_count_sync(content)
    
    def get_word_count_sync(self, text: str) -> int:
        """Get word count from text (synchronous)."""
        if not text:
            return 0
        return len(text.split())
    
    async def is_empty(self) -> bool:
        """Check if paper is empty."""
        content = await self.get_paper()
        return len(content.strip()) == 0
    
    def set_rechunk_callback(self, callback: Callable) -> None:
        """Set callback to trigger re-chunking when paper updated."""
        self.rechunk_callback = callback
    
    def get_version(self) -> int:
        """Get current paper version."""
        return self.version
    
    async def initialize_with_placeholders(self, first_body_content: str) -> None:
        """
        Initialize paper with first body content and all section placeholders.
        Called when the first body section is accepted.
        
        Args:
            first_body_content: The first body section content (already validated)
        """
        # Build paper with placeholders framing the body content
        paper = (
            f"{ABSTRACT_PLACEHOLDER}\n\n"
            f"{INTRO_PLACEHOLDER}\n\n"
            f"{first_body_content}\n\n"
            f"{CONCLUSION_PLACEHOLDER}\n\n"
            f"{PAPER_ANCHOR}"
        )
        
        async with self._lock:
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(paper)
            
            self.version += 1
            logger.info(f"Paper initialized with placeholders (version {self.version})")
        
        # Trigger re-chunking callback
        if self.rechunk_callback:
            try:
                await self.rechunk_callback(paper)
            except Exception as e:
                logger.error(f"Re-chunking callback failed: {e}")
    
    async def replace_placeholder(self, placeholder: str, content: str) -> bool:
        """
        Replace a placeholder with validated content.
        Each placeholder is replaced exactly once.
        
        Args:
            placeholder: The placeholder constant to replace
            content: The validated section content
        
        Returns:
            True if placeholder was found and replaced, False otherwise
        """
        async with self._lock:
            if not self.file_path.exists():
                logger.warning("Cannot replace placeholder: paper file does not exist")
                return False
            
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                paper = await f.read()
            
            if placeholder not in paper:
                logger.warning(f"Placeholder not found in paper: {placeholder[:50]}...")
                return False
            
            # Replace the placeholder with the content (only once)
            new_paper = paper.replace(placeholder, content, 1)
            
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(new_paper)
            
            self.version += 1
            logger.info(f"Placeholder replaced (version {self.version})")
        
        # Trigger re-chunking callback OUTSIDE the lock
        if self.rechunk_callback:
            try:
                await self.rechunk_callback(new_paper)
            except Exception as e:
                logger.error(f"Re-chunking callback failed: {e}")
        
        return True
    
    def has_placeholder(self, placeholder: str, paper_content: str) -> bool:
        """
        Check if a placeholder exists in the paper content.
        
        Args:
            placeholder: The placeholder constant to check
            paper_content: The current paper content
        
        Returns:
            True if placeholder is present, False otherwise
        """
        return placeholder in paper_content
    
    async def clear_body_section(self) -> None:
        """
        Clear body section but preserve placeholders.
        Removes all content between the first body content and CONCLUSION_PLACEHOLDER.
        Keeps Abstract, Introduction, and Conclusion placeholders intact.
        """
        final_content = None
        
        async with self._lock:
            # Use unlocked version since we already hold the lock
            current_paper = await self._get_paper_unlocked()
            
            if not current_paper:
                logger.warning("Cannot clear body - paper is empty")
                return
            
            # Find where body content starts (after Abstract and Introduction placeholders)
            lines = current_paper.split('\n')
            body_start_idx = None
            body_end_idx = None
            
            # Find first non-placeholder, non-anchor line (this is body content)
            for i, line in enumerate(lines):
                if (line.strip() and 
                    ABSTRACT_PLACEHOLDER not in line and
                    INTRO_PLACEHOLDER not in line and
                    CONCLUSION_PLACEHOLDER not in line and
                    PAPER_ANCHOR not in line):
                    body_start_idx = i
                    break
            
            # Find where body ends (at CONCLUSION_PLACEHOLDER or PAPER_ANCHOR)
            for i, line in enumerate(lines):
                if CONCLUSION_PLACEHOLDER in line or PAPER_ANCHOR in line:
                    body_end_idx = i
                    break
            
            if body_start_idx is None:
                logger.info("No body content found to clear")
                return
            
            # Reconstruct paper with body section removed
            new_lines = []
            
            # Keep placeholders before body
            new_lines.extend(lines[:body_start_idx])
            
            # Skip body content, add everything from conclusion placeholder onward
            if body_end_idx is not None:
                new_lines.extend(lines[body_end_idx:])
            
            new_paper = '\n'.join(new_lines)
            
            # Save using unlocked version since we already hold the lock
            final_content = await self._update_paper_unlocked(new_paper)
            
            logger.info(f"Cleared body section (lines {body_start_idx} to {body_end_idx})")
        
        # Trigger re-chunking callback OUTSIDE the lock to avoid deadlock
        if final_content and self.rechunk_callback:
            try:
                await self.rechunk_callback(final_content)
            except Exception as e:
                logger.error(f"Re-chunking callback failed after clearing body: {e}")
    
    async def store_previous_version(
        self,
        version: int,
        title: str,
        body: str,
        critique_feedback: str
    ) -> None:
        """
        Store previous body version for UI display.
        
        Args:
            version: Version number
            title: Paper title for this version
            body: Body section content
            critique_feedback: Critique feedback that triggered rewrite
        """
        async with self._lock:
            # Add to in-memory list
            version_data = {
                "version": version,
                "title": title,
                "body": body,
                "critique_feedback": critique_feedback
            }
            self.previous_versions.append(version_data)
            
            # Save to file
            version_file = Path(system_config.data_dir) / f"paper_version_{version}.txt"
            version_file.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(version_file, 'w', encoding='utf-8') as f:
                await f.write(f"VERSION {version}: {title}\n")
                await f.write(f"{'=' * 80}\n\n")
                await f.write(f"BODY SECTION:\n{body}\n\n")
                await f.write(f"{'=' * 80}\n\n")
                await f.write(f"CRITIQUE FEEDBACK THAT TRIGGERED REWRITE:\n{critique_feedback}\n")
            
            logger.info(f"Stored previous version {version} to {version_file}")
    
    async def get_previous_versions(self) -> list:
        """
        Get all previous versions for UI display.
        
        Returns:
            List of version dicts with version, title, body, critique_feedback
        """
        async with self._lock:
            return self.previous_versions.copy()
    
    async def ensure_placeholders_exist(self) -> bool:
        """
        Ensure placeholders exist in the paper when resuming from an existing file.
        
        Papers created before the placeholder system or from older code versions
        may not have placeholders. This causes "old_string not found" failures
        when the model tries to use placeholder text as old_string.
        
        This method:
        1. Checks if ABSTRACT_PLACEHOLDER, INTRO_PLACEHOLDER, CONCLUSION_PLACEHOLDER exist
        2. If any are missing, attempts to reconstruct the paper with placeholders
        3. Only adds MISSING placeholders - doesn't duplicate existing ones
        
        Returns:
            True if placeholders were added, False if they already existed
        """
        async with self._lock:
            if not self.file_path.exists():
                logger.warning("Cannot ensure placeholders: paper file does not exist")
                return False
            
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                paper = await f.read()
            
            if not paper.strip():
                logger.info("Paper is empty - no placeholders needed")
                return False
            
            # Check which placeholders are missing
            has_abstract_placeholder = ABSTRACT_PLACEHOLDER in paper
            has_intro_placeholder = INTRO_PLACEHOLDER in paper
            has_conclusion_placeholder = CONCLUSION_PLACEHOLDER in paper
            has_anchor = PAPER_ANCHOR in paper
            
            # Check for actual section content (not placeholders)
            # Use flexible patterns to detect if sections have been written
            import re
            has_abstract_content = bool(re.search(r'^Abstract\s*$', paper, re.IGNORECASE | re.MULTILINE))
            has_intro_content = bool(re.search(r'^I\.?\s+Introduction|^Introduction\s*$', paper, re.IGNORECASE | re.MULTILINE))
            has_conclusion_content = bool(re.search(r'^(?:IX|VIII|VII|VI|V)\.?\s+Conclusion|^\d+\.?\s+Conclusion|^Conclusion\s*$', paper, re.IGNORECASE | re.MULTILINE))
            
            # If all placeholders exist OR corresponding content exists, nothing to do
            if (has_abstract_placeholder or has_abstract_content) and \
               (has_intro_placeholder or has_intro_content) and \
               (has_conclusion_placeholder or has_conclusion_content):
                logger.info("Placeholders check: All sections either have placeholders or actual content")
                return False
            
            logger.info(f"Placeholder check - abstract: {has_abstract_placeholder}/{has_abstract_content}, "
                       f"intro: {has_intro_placeholder}/{has_intro_content}, "
                       f"conclusion: {has_conclusion_placeholder}/{has_conclusion_content}")
            
            # Need to add missing placeholders
            # Extract current body content (everything that's not a placeholder or anchor)
            lines = paper.split('\n')
            body_lines = []
            
            for line in lines:
                # Skip existing placeholders and anchor
                if ABSTRACT_PLACEHOLDER in line or \
                   INTRO_PLACEHOLDER in line or \
                   CONCLUSION_PLACEHOLDER in line or \
                   PAPER_ANCHOR in line:
                    continue
                body_lines.append(line)
            
            body_content = '\n'.join(body_lines).strip()
            
            if not body_content:
                logger.warning("No body content found - cannot add placeholders to empty paper")
                return False
            
            # Reconstruct paper with placeholders for missing sections
            # Only add placeholder if:
            # 1. The placeholder doesn't already exist, AND
            # 2. The actual content doesn't exist either
            new_paper_parts = []
            
            # Abstract at top
            if not has_abstract_placeholder and not has_abstract_content:
                new_paper_parts.append(ABSTRACT_PLACEHOLDER)
                new_paper_parts.append("")
            
            # Introduction after abstract
            if not has_intro_placeholder and not has_intro_content:
                new_paper_parts.append(INTRO_PLACEHOLDER)
                new_paper_parts.append("")
            
            # Body content
            new_paper_parts.append(body_content)
            new_paper_parts.append("")
            
            # Conclusion before anchor
            if not has_conclusion_placeholder and not has_conclusion_content:
                new_paper_parts.append(CONCLUSION_PLACEHOLDER)
                new_paper_parts.append("")
            
            # Anchor at end
            new_paper_parts.append(PAPER_ANCHOR)
            
            new_paper = '\n'.join(new_paper_parts)
            
            # Write updated paper
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(new_paper)
            
            self.version += 1
            logger.info(f"Added missing placeholders to paper (version {self.version})")
            
            return True


# Global paper memory instance
paper_memory = PaperMemory()

