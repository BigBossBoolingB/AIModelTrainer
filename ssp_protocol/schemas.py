# ssp_protocol/schemas.py

"""
Defines the data structures (schemas) used to pass information
between the modules of the Syntactic-Semantic-Pragmatic (SSP) Protocol.
"""

from dataclasses import dataclass, field
from typing import Any, List, Dict

@dataclass
class CreativeDirective:
    """Input: The initial goal and constraints for the generation task."""
    goal_id: str
    subject_matter: str
    tone: str
    audience: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RawLinguisticVector:
    """Output of the Poet: A raw, creative exploration of the subject."""
    goal_id: str
    vector_content: str
    associated_metadata: Dict[str, Any] = field(default_factory=dict)
    divergence_score: float = 0.0

@dataclass
class ValidatedVector:
    """Output of the Logician: A fact-checked and refined vector."""
    goal_id: str
    vector_content: str
    validation_log: List[str] = field(default_factory=list)
    coherence_score: float = 0.0

@dataclass
class PragmaticallyOptimizedPayload:
    """Output of the Diplomat: The final, structured, and audience-optimized text."""
    goal_id: str
    final_text: str
    optimization_log: List[str] = field(default_factory=list)
    final_confidence_score: float = 0.0
