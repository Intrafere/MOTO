"""
Rigor prompts for mathematical rigor enhancement.
"""

from backend.compiler.memory.compiler_rejection_log import compiler_rejection_log


def get_rigor_system_prompt() -> str:
    """Get system prompt for mathematical rigor enhancement mode."""
    return """You are enhancing the mathematical rigor of a mathematical document. Your role is to:

1. Review the current document
2. Review the outline
3. Identify areas where mathematical rigor can be improved
4. Propose specific enhancements

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
Identify where rigor can be improved - or if the document already maintains appropriate mathematical rigor for its current stage of construction, state such instead of creating an enhancement attempt and note your refusal to make changes as they would be unnecessary or could degrade readability.

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

The CURRENT DOCUMENT you are reviewing may contain system-managed markers:

**SECTION PLACEHOLDERS** (show where sections will be written):
- [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION...]
- [HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION...]
- [HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION...]

**PAPER ANCHOR** (marks document boundary):
- [HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]

IMPORTANT: These markers are SYSTEM-MANAGED (added by paper_memory.py), NOT AI-generated content. They are NORMAL and EXPECTED during document construction.

**YOU MUST NEVER OUTPUT THESE MARKERS IN YOUR ENHANCEMENTS**

When proposing enhancements:
- Do NOT include any of these markers in your enhancement content
- Placeholders in the current document are expected - don't try to remove them
- Your enhancements should contain only actual mathematical prose

Consider these areas for potential improvement:
- Precision of mathematical claims and definitions
- Completeness of proofs and logical arguments
- Justification and supporting lemmas
- Acknowledgment of assumptions and edge cases
- Clarity of mathematical notation
- Addressing potential counterexamples or boundary conditions

If enhancements are warranted, propose revisions that strengthen mathematical grounding while preserving correctness and flow.

RIGOR ENHANCEMENTS INCLUDE:
- Strengthening proof arguments with additional steps
- Clarifying assumptions and conditions
- Adding intermediate lemmas where appropriate
- Improving precision of mathematical language and notation
- Addressing potential counterexamples or edge cases
- Strengthening logical connections between claims
- Ensuring all claims are properly justified
- Adding clarifying remarks for complex arguments

CRITICAL REQUIREMENTS:
- Only propose enhancements that ADD mathematical rigor, not change content
- Maintain the existing narrative and structure
- Don't remove or substantially alter existing content
- Focus on making mathematical claims more precise and defensible
- DO NOT add forward-looking structural previews
- Add rigor to existing mathematical prose, not roadmaps of future content
- Focus on sound mathematical reasoning

EXACT STRING MATCHING FOR EDITS:
This system uses EXACT STRING MATCHING. To make an enhancement, you must:
1. Identify the EXACT text in the current document where you want to add/modify
2. Copy that exact text (including whitespace and newlines) as old_string
3. Provide your enhanced version as new_string
4. Choose the appropriate operation (replace, insert_after)

OPERATIONS:
- "replace": Find old_string exactly, replace it with new_string (for rewriting text with more rigor)
- "insert_after": Find old_string exactly (as anchor), insert new_string after it (for adding lemmas/remarks)

UNIQUENESS REQUIREMENT:
- old_string MUST be unique in the document
- If the text you want to modify appears multiple times, include MORE surrounding context
- Include enough text (typically 3-5 lines) to ensure uniqueness

If the document is already rigorous enough at this stage of construction, you may respond that no enhancement is needed.

The validator will reject if:
- Enhancement reduces rigor or clarity
- old_string is not found or not unique
- Changes undermine existing coherence
- Enhancement is inappropriate for current draft stage
- Enhancement introduces unsound mathematical claims

Output your response ONLY as JSON in this exact format:
{
  "needs_enhancement": true or false,
  "operation": "replace | insert_after",
  "old_string": "exact text from document to find (empty if needs_enhancement=false)",
  "new_string": "enhanced text (empty if needs_enhancement=false)",
  "reasoning": "Why this enhancement is or isn't needed"
}
"""


def get_rigor_json_schema() -> str:
    """Get JSON schema specification for rigor mode."""
    return """
REQUIRED JSON FORMAT:
{
  "needs_enhancement": true OR false,
  "operation": "replace | insert_after",
  "old_string": "exact text from document to find (empty if needs_enhancement=false)",
  "new_string": "enhanced text (empty if needs_enhancement=false)",
  "reasoning": "string - explanation of why enhancement is or isn't needed"
}

FIELD DEFINITIONS:
- needs_enhancement: Whether a rigor enhancement should be made
- operation: Type of edit operation:
  * "replace": Find old_string and replace with new_string (for rewriting with more rigor)
  * "insert_after": Find old_string (anchor) and insert new_string after it (for adding lemmas/remarks)
- old_string: EXACT text from the current document that you want to modify or use as anchor.
  Must be unique. Include enough context (3-5 lines) to ensure uniqueness. Empty if needs_enhancement=false.
- new_string: The enhanced text. Empty if needs_enhancement=false.
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

Example (Enhancement needed - inserting after anchor):
{
  "needs_enhancement": true,
  "operation": "insert_after",
  "old_string": "Theorem 2.3: A number \\\\alpha is constructible if and only if\\nit lies in a field extension of \\\\mathbb{Q} of degree 2^n for some n \\\\geq 0.",
  "new_string": "\\n\\nWe note that this characterization requires the field extension to be normal and separable over \\\\mathbb{Q}. Specifically, if K/\\\\mathbb{Q} is a field extension containing a constructible number \\\\alpha, then there exists a tower \\\\mathbb{Q} = K_0 \\\\subset K_1 \\\\subset \\\\ldots \\\\subset K_n = K where each K_{i+1}/K_i has degree exactly 2.",
  "reasoning": "The theorem statement in Section II lacks the precise field-theoretic conditions for constructibility. Adding this remark strengthens the mathematical foundation and makes the subsequent proof more accessible."
}

Example (No enhancement needed):
{
  "needs_enhancement": false,
  "operation": "replace",
  "old_string": "",
  "new_string": "",
  "reasoning": "The document maintains appropriate mathematical rigor for its current stage. All theorems have complete proofs, definitions are precise, and logical connections between claims are clear. Further enhancement at this point could reduce readability without significant benefit."
}

Example (Adding a lemma - inserting after anchor):
{
  "needs_enhancement": true,
  "operation": "insert_after",
  "old_string": "This completes the proof of Theorem 4.1. \\\\square\\n\\nWe now proceed to our main result.",
  "new_string": "\\n\\nLemma 4.2: If \\\\alpha is algebraic over \\\\mathbb{Q} with minimal polynomial of degree n, then [\\\\mathbb{Q}(\\\\alpha):\\\\mathbb{Q}] = n.\\n\\nProof: By definition of the minimal polynomial, it is the unique monic polynomial of smallest degree that has \\\\alpha as a root. The elements 1, \\\\alpha, \\\\alpha^2, \\\\ldots, \\\\alpha^{n-1} form a basis for \\\\mathbb{Q}(\\\\alpha) over \\\\mathbb{Q}, as any higher power of \\\\alpha can be expressed as a linear combination of these using the minimal polynomial relation. \\\\square",
  "reasoning": "Theorem 4.3 uses the relationship between minimal polynomial degree and field extension degree without explicit justification. Adding this lemma makes the argument self-contained and strengthens the logical chain."
}
"""


async def build_rigor_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str
) -> str:
    """
    Build complete prompt for rigor enhancement mode.
    
    Note: Aggregator database is NOT included in rigor mode context.
    Outline is ALWAYS fully injected (never RAGed).
    
    Args:
        user_prompt: User's compiler-directing prompt
        current_outline: Current outline (ALWAYS fully injected)
        current_paper: Current document (may be RAG-retrieved if large)
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_rigor_system_prompt(),
        "\n---\n",
        get_rigor_json_schema(),
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
        "Now decide if rigor enhancement is needed (respond as JSON):"
    ])
    
    return "\n".join(parts)
