"""
Tier 3: Final Answer Agents

Agents responsible for synthesizing research papers into a final answer
to the user's original research question.

Key Agents:
- CertaintyAssessor: Assesses what can be answered with certainty from papers
- AnswerFormatSelector: Chooses short form (single paper) or long form (volume)
- VolumeOrganizer: Organizes volume structure for long form answers

CRITICAL: These agents operate ONLY on Tier 2 papers, NOT on Tier 1 brainstorm databases.
"""
from backend.autonomous.agents.final_answer.certainty_assessor import CertaintyAssessor
from backend.autonomous.agents.final_answer.answer_format_selector import AnswerFormatSelector
from backend.autonomous.agents.final_answer.volume_organizer import VolumeOrganizer

__all__ = [
    "CertaintyAssessor",
    "AnswerFormatSelector", 
    "VolumeOrganizer"
]

