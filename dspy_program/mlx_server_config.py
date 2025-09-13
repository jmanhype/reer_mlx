"""MLX Server configuration for DSPy integration.

Start the MLX server with:
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080

This provides an OpenAI-compatible API that DSPy can use natively.
"""

import dspy
import logging

logger = logging.getLogger(__name__)


def configure_mlx_server(
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit", port: int = 8080
):
    """Configure DSPy to use MLX server with OpenAI-compatible API.

    The 3B model is required for proper DSPy structured output support.
    The 1B model cannot follow DSPy's output format instructions.

    Returns:
        True if configuration successful, False otherwise
    """
    try:
        # Create LM with OpenAI-compatible endpoint
        lm = dspy.LM(
            model=f"openai/{model}",
            api_base=f"http://localhost:{port}/v1/",
            api_key="not-needed",  # MLX server doesn't require auth
        )

        # Configure DSPy globally
        dspy.configure(lm=lm)

        logger.info(f"✓ DSPy configured with MLX server (3B model) at port {port}")
        return True

    except Exception as e:
        logger.error(f"Failed to configure MLX server: {e}")
        return False


def test_mlx_dspy_integration():
    """Test that MLX server + DSPy integration works."""

    # Define a simple signature
    class TestSignature(dspy.Signature):
        """Test signature for MLX-DSPy integration."""

        input_text = dspy.InputField(desc="Input text")
        output_text = dspy.OutputField(desc="Output text")

    try:
        # Configure MLX server
        if not configure_mlx_server():
            return False, "MLX server not running"

        # Create predictor
        predictor = dspy.Predict(TestSignature)

        # Test prediction
        result = predictor(input_text="Hello world")

        if hasattr(result, "output_text"):
            return True, f"Success: {result.output_text}"
        else:
            return False, "No output_text in result"

    except Exception as e:
        return False, f"Error: {e}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing MLX Server + DSPy Integration...")
    print("Make sure MLX server is running:")
    print("mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080")
    print()

    success, message = test_mlx_dspy_integration()
    if success:
        print(f"✅ Integration working: {message}")
    else:
        print(f"❌ Integration failed: {message}")
