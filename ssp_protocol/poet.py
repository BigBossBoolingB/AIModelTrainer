# ssp_protocol/poet.py

from .schemas import CreativeDirective, RawLinguisticVector

def generate_raw_vector(directive: CreativeDirective) -> RawLinguisticVector:
    """
    Core L1: The Poet.
    Takes a directive and generates a raw creative output.
    (Pass-through prototype version)
    """
    print(f"POET: Received directive for goal '{directive.goal_id}'. Generating raw vector...")

    # In a real implementation, this would involve complex LLM calls.
    # Here, we just return a hard-coded sample.
    return RawLinguisticVector(
        goal_id=directive.goal_id,
        vector_content="This is the initial creative and divergent text from the Poet, exploring the subject of " + directive.subject_matter,
        associated_metadata={'source': 'Poet Prototype v0.1'},
        divergence_score=0.85
    )
