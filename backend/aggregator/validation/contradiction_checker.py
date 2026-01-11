"""
Contradiction checker - detects logical contradictions in submissions.
"""
import re
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ContradictionChecker:
    """Detects contradictions in submission text."""
    
    def __init__(self):
        self.contradiction_keywords = [
            "contradicts",
            "conflicts with",
            "does not support",
            "inconsistent with",
            "opposes",
            "disputes"
        ]
        
        self.negation_words = ["not", "no", "never", "none", "neither", "cannot", "can't"]
    
    def check_contradictions(
        self,
        text: str,
        context: str = ""
    ) -> Tuple[bool, str]:
        """
        Check for contradictions in text.
        
        Args:
            text: Text to check
            context: Optional context (e.g., previous submissions)
        
        Returns:
            (passed, reason) - True if no contradictions, False with reason if found
        """
        # Check for explicit contradiction keywords
        text_lower = text.lower()
        
        for keyword in self.contradiction_keywords:
            if keyword in text_lower:
                return False, f"Explicit contradiction detected: contains '{keyword}'"
        
        # Check for negation patterns that might indicate contradiction
        # e.g., "the evidence does not support" or "this is not correct"
        negation_patterns = [
            r'\b(evidence|data|research)\s+(does not|doesn\'t|cannot|can\'t)\s+support',
            r'\b(this|that)\s+is\s+not\s+(correct|accurate|true|valid)',
            r'\bnot\s+supported\s+by',
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, text_lower):
                return False, f"Potential contradiction detected: negation pattern '{pattern}'"
        
        return True, "No contradictions detected"


# Global contradiction checker instance
contradiction_checker = ContradictionChecker()

