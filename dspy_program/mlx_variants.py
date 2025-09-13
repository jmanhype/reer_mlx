"""MLX-based variant generation for REER."""

import logging
import mlx_lm

logger = logging.getLogger(__name__)

# Global model instance
_model = None
_tokenizer = None


def get_mlx_model():
    """Get or load the MLX model."""
    global _model, _tokenizer
    if _model is None:
        logger.info("Loading MLX model for variant generation...")
        _model, _tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        logger.info("MLX model loaded successfully")
    return _model, _tokenizer


def generate_variant_with_mlx(segment: str, style: str) -> str:
    """Generate a variant using MLX directly."""
    model, tokenizer = get_mlx_model()

    # Create a prompt that encourages rephrasing
    prompt = f"""Rephrase the following text to be {style}:
Original: {segment}
Rephrased:"""

    try:
        response = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=100, verbose=False
        )

        # Extract the rephrased part
        if "Rephrased:" in response:
            rephrased = response.split("Rephrased:")[-1].strip()
        else:
            rephrased = response.strip()

        # Clean up the response
        rephrased = rephrased.split("\n")[0].strip()  # Take first line only

        return rephrased
    except Exception as e:
        logger.error(f"MLX generation failed: {e}")
        return ""


def propose_segment_variants_mlx(segment: str, k: int = 2) -> list[str]:
    """Generate k variants using MLX."""
    styles = [
        "more systematic and methodical",
        "more specific with concrete steps",
        "more actionable and measurable",
        "clearer and more precise",
        "more detailed and comprehensive",
    ]

    variants = []
    for i, style in enumerate(styles[:k]):
        variant = generate_variant_with_mlx(segment, style)
        if variant and variant != segment:
            variants.append(variant)
            logger.debug(f"Generated variant {i+1}: {variant[:50]}...")

    return variants
