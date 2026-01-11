"""
Review prompts for mathematical document cleanup and error correction.
"""

from backend.compiler.memory.compiler_rejection_log import compiler_rejection_log


def get_review_system_prompt() -> str:
    """Get system prompt for document review/cleanup mode."""
    return """You are reviewing the current mathematical document draft for errors and needed improvements. Your role is to:

1. Review ONLY the current document (aggregator database is NOT in your context for this task)
2. Identify any obvious errors or issues
3. Decide if an edit is needed or if the document is acceptable as a "draft in progress"

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

YOUR TASK:
Review the document for these specific issues:
- Grammar errors
- Clarity problems
- Mathematical accuracy issues
- Logical errors or gaps
- Structural issues
- Redundancy
- Forward-looking structural previews
- Other improvements

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

The CURRENT DOCUMENT you are reviewing may contain system-managed markers:

**SECTION PLACEHOLDERS** (show where sections will be written):
- [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION...]
- [HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION...]
- [HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION...]

**PAPER ANCHOR** (marks document boundary):
- [HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]

IMPORTANT: These markers are SYSTEM-MANAGED (added by paper_memory.py), NOT AI-generated content. They are NORMAL and EXPECTED during document construction.

**YOU MUST NEVER OUTPUT THESE MARKERS IN YOUR EDITS**

When making edits:
- Do NOT include any of these markers in your edit content
- Placeholders in the current document are expected - don't try to remove them
- Your edits should contain only actual mathematical prose

WHEN TO MAKE AN EDIT:
- Clear grammatical errors
- Obvious redundancy that should be removed
- Coherence issues between sections
- Terminology inconsistencies
- Mathematical inaccuracies or logical errors
- Significant clarity improvements possible
- Forward-looking structural language outside introduction (e.g., 'Section III will...', bulleted lists of future content)
- Unfounded claims or logical fallacies that should be corrected

WHEN NOT TO MAKE AN EDIT:
- Document is acceptable for a draft in progress
- Only minor stylistic preferences
- Changes would be purely cosmetic
- No obvious issues found

EXACT STRING MATCHING FOR EDITS:
This system uses EXACT STRING MATCHING. To make an edit, you must:
1. Identify the EXACT text in the current document that you want to modify
2. Copy that exact text (including whitespace and newlines) as old_string
3. Provide your corrected/improved version as new_string
4. Choose the appropriate operation (replace, insert_after, delete)

OPERATIONS:
- "replace": Find old_string exactly, replace it with new_string
- "insert_after": Find old_string exactly (as anchor), insert new_string after it
- "delete": Find old_string exactly, remove it (new_string should be empty)

UNIQUENESS REQUIREMENT:
- old_string MUST be unique in the document
- If the text you want to edit appears multiple times, include MORE surrounding context
- Include enough text (typically 3-5 lines) to ensure uniqueness

If NO edit is needed, set "needs_edit" to false and leave old_string and new_string empty.

Output your response ONLY as JSON in this exact format:
{
  "needs_edit": true or false,
  "operation": "replace | insert_after | delete",
  "old_string": "exact text from document to find (empty if needs_edit=false)",
  "new_string": "corrected text or text to insert (empty if delete or needs_edit=false)",
  "reasoning": "Why edit is or isn't needed"
}
"""


def get_review_json_schema() -> str:
    """Get JSON schema specification for review mode."""
    return """
REQUIRED JSON FORMAT:
{
  "needs_edit": true OR false,
  "operation": "replace | insert_after | delete",
  "old_string": "exact text from document to find (empty if needs_edit=false)",
  "new_string": "corrected text or text to insert (empty if delete or needs_edit=false)",
  "reasoning": "string - explanation of why edit is or isn't needed"
}

FIELD DEFINITIONS:
- needs_edit: Whether an edit should be made to the document
- operation: Type of edit operation:
  * "replace": Find old_string and replace with new_string
  * "insert_after": Find old_string (anchor) and insert new_string after it
  * "delete": Find old_string and remove it (new_string should be empty)
- old_string: EXACT text from the current document that you want to modify. Must be unique.
  Include enough context (3-5 lines) to ensure uniqueness. Empty if needs_edit=false.
- new_string: The corrected/improved text (for replace/insert_after). Empty for delete or needs_edit=false.
- reasoning: Explain your decision

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau", write "\\\\(" not "\\("
2. Quotes: Escape double quotes inside strings as \\"
   - Example: "He said \\"hello\\"" 
3. Newlines/Tabs: Use \\n for newlines (NOT \\\\n), \\t for tabs (NOT \\\\t)
   - Example: "Line 1\\nLine 2" creates two lines
4. DO NOT use single backslashes except for: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
5. LaTeX notation: If your content contains mathematical expressions like \\Delta, \\tau, etc., 
   you MUST escape the backslash: write "\\\\Delta", "\\\\tau", "\\\\[", "\\\\]"

Example (Replace - fixing an error):
{
  "needs_edit": true,
  "operation": "replace",
  "old_string": "Since \\\\pi is algebraic over \\\\mathbb{Q}, the expression \\\\pi^2 + 1\\nmust also be algebraic. This follows from basic field theory.",
  "new_string": "Since \\\\pi is transcendental over \\\\mathbb{Q}, any algebraic expression involving \\\\pi must also be transcendental. Therefore, \\\\sqrt{\\\\pi} cannot be constructible.",
  "reasoning": "Section III contains a mathematical error stating that pi is algebraic. This contradicts the Lindemann-Weierstrass theorem established earlier. The correction clarifies the transcendental nature of pi."
}

Example (No edit needed):
{
  "needs_edit": false,
  "operation": "replace",
  "old_string": "",
  "new_string": "",
  "reasoning": "The document is coherent and mathematically accurate for its current stage of construction. All proofs are logically sound and definitions are properly introduced before use. No immediate corrections required."
}

Example (Deletion - removing redundant content):
{
  "needs_edit": true,
  "operation": "delete",
  "old_string": "A transcendental number is a real or complex number that is not algebraic,\\nmeaning it is not the root of any non-zero polynomial equation with rational\\ncoefficients. This definition was previously stated in Section II.C.",
  "new_string": "",
  "reasoning": "The paragraph in Section IV restates the definition of transcendental numbers already given in Section II.C without adding new information. Removing it improves document flow."
}
"""


async def build_review_prompt(
    user_prompt: str,
    current_paper: str,
    current_outline: str
) -> str:
    """
    Build complete prompt for review mode.
    
    Note: Aggregator database is NOT included in review mode context.
    
    Args:
        user_prompt: User's compiler-directing prompt
        current_paper: Current document to review
        current_outline: Current outline for structural reference (always fully injected)
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_review_system_prompt(),
        "\n---\n",
        get_review_json_schema(),
        "\n---\n"
    ]
    
    # Add rejection history (DIRECT INJECTION - almost always fits)
    rejection_history = await compiler_rejection_log.get_rejections_text()
    if rejection_history:
        parts.append(f"""YOUR RECENT REJECTION HISTORY (Last 10 rejections):
{rejection_history}

LEARN FROM THESE PAST MISTAKES.
---
""")
    
    parts.extend([
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"CURRENT DOCUMENT:\n{current_paper}",
        "\n---\n",
        "Now review the document and decide if an edit is needed (respond as JSON):"
    ])
    
    return "\n".join(parts)
