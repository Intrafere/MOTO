"""
Paper Critique Prompts Module.

Contains the default critique prompt and helper functions for building
critique requests to the validator model.
"""

# Default critique prompt that can be customized by users
DEFAULT_CRITIQUE_PROMPT = """You are an expert academic reviewer providing an honest, thorough critique of a research paper.

Evaluate this paper and provide:
1. NOVELTY (1-10): How original and innovative is this work?
2. CORRECTNESS (1-10): How mathematically/logically sound is the content?
3. IMPACT ON RELATED FIELD (1-10): How significant could this contribution be?

For each category, provide the numeric rating (1-10) and detailed feedback explaining your assessment.

Be honest and constructive. Identify both strengths and weaknesses."""


# JSON schema for structured output (always appended, not customizable)
CRITIQUE_JSON_SCHEMA = """
OUTPUT FORMAT (respond ONLY with valid JSON):
{
  "novelty_rating": <integer 1-10>,
  "novelty_feedback": "<detailed feedback on novelty>",
  "correctness_rating": <integer 1-10>,
  "correctness_feedback": "<detailed feedback on correctness>",
  "impact_rating": <integer 1-10>,
  "impact_feedback": "<detailed feedback on potential impact>",
  "full_critique": "<comprehensive summary critique of the paper>"
}

IMPORTANT:
- All ratings MUST be integers from 1 to 10
- All feedback fields MUST be non-empty strings
- Respond ONLY with the JSON object, no additional text
"""


def build_critique_prompt(paper_content: str, paper_title: str = None, custom_prompt: str = None) -> str:
    """
    Build the complete critique prompt for the validator.
    
    Args:
        paper_content: The full paper content to critique
        paper_title: Optional title of the paper being critiqued
        custom_prompt: Optional custom prompt to use instead of default
        
    Returns:
        The complete prompt string to send to the validator
    """
    # Use custom prompt if provided, otherwise use default
    base_prompt = custom_prompt if custom_prompt else DEFAULT_CRITIQUE_PROMPT
    
    # Build title section if provided
    title_section = f"\nPAPER TITLE: {paper_title}\n" if paper_title else ""
    
    # Build the complete prompt
    complete_prompt = f"""{base_prompt}

{CRITIQUE_JSON_SCHEMA}

---
PAPER TO REVIEW:{title_section}
---

{paper_content}

---
END OF PAPER
---

Now provide your honest critique as JSON:"""
    
    return complete_prompt


def get_default_critique_prompt() -> str:
    """
    Get the default critique prompt text.
    
    Returns:
        The default critique prompt string (without JSON schema)
    """
    return DEFAULT_CRITIQUE_PROMPT

