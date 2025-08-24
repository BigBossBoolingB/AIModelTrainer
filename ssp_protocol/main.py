# ssp_protocol/main.py

from .schemas import CreativeDirective
from .poet import generate_raw_vector
from .logician import validate_vector
from .diplomat import structure_and_assess

def run_ssp_protocol_prototype():
    """
    Orchestrates the full end-to-end flow of the SSP Protocol prototype.
    """
    print("--- SSP PROTOCOL PROTOTYPE: START ---")

    # 1. Define the initial creative goal
    directive = CreativeDirective(
        goal_id="PRJ-001-Q3-Marketing-Campaign",
        subject_matter="The launch of our new flagship product, 'The Catalyst'.",
        tone="Excited and professional"
    )
    print(f"\n[DIRECTIVE] Goal: {directive.goal_id}, Subject: '{directive.subject_matter}'")
    print("-" * 20)

    # 2. Pass to the Poet for creative generation
    raw_vector = generate_raw_vector(directive)
    print(f"POET OUTPUT: '{raw_vector.vector_content}' (Divergence: {raw_vector.divergence_score})")
    print("-" * 20)

    # 3. Pass to the Logician for validation and refinement
    validated_vector = validate_vector(raw_vector)
    print(f"LOGICIAN OUTPUT: '{validated_vector.vector_content}' (Coherence: {validated_vector.coherence_score})")
    print("-" * 20)

    # 4. Pass to the Diplomat for pragmatic structuring and assessment
    final_payload = structure_and_assess(validated_vector)
    print(f"DIPLOMAT OUTPUT (Final Payload):")
    print("=" * 40)
    print(final_payload.final_text)
    print("=" * 40)
    print(f"Final Confidence Score: {final_payload.final_confidence_score}")

    print("\n--- SSP PROTOCOL PROTOTYPE: END ---")


if __name__ == "__main__":
    run_ssp_protocol_prototype()
