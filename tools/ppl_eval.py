"""Perplexity evaluators for REER: MLX (local) and Together via dspy.LM.

This module intentionally supports only two backends:

- MLX local models: compute conditional PPL(y|x,z) by masking context tokens.
- Together via dspy.LM: uses a client's loglikelihood API if available.

All heuristic/proxy evaluators have been removed by design.
"""

from __future__ import annotations

from collections.abc import Callable
import math


def make_mlx_ppl_evaluator(
    model_name: str, window_size: int = 2048, stride: int = 1024
) -> Callable[[str, str, str], float]:
    """Create a conditional PPL(y|x,z) evaluator using MLX locally.

    Implementation notes:
    - Concatenate: [BOS] + x + \n\n + z + \n\n + y
    - Forward once; compute log_softmax over logits
    - Accumulate log-likelihood over y tokens only (mask-out x+z region)
    - For long contexts exceeding window_size:
      * Use sliding windows with overlapping stride
      * Average log-probabilities across windows for overlapping tokens
      * Ensure y tokens are always fully included in at least one window

    Args:
        model_name: MLX model identifier
        window_size: Maximum context window (default 2048)
        stride: Sliding window stride for long contexts (default 1024)
    """
    try:
        import mlx.core as mx  # type: ignore
        from mlx_lm import load  # type: ignore

        model, tokenizer = load(model_name)

        def ppl_fn(x: str, y: str, z: str) -> float:
            if not y:
                return 10.0

            # Build combined token sequence
            def enc(s: str) -> list[int]:
                return tokenizer.encode(s)

            sep = "\n\n"
            ids_x = enc(x)
            ids_z = enc(z)
            ids_y = enc(y)

            bos = enc(tokenizer.bos_token) if hasattr(tokenizer, "bos_token") else []
            full = (bos or []) + ids_x + enc(sep) + ids_z + enc(sep) + ids_y

            # Determine y region indices
            y_start = len(full) - len(ids_y)

            # Simple case: fits in single window
            if len(full) <= window_size:
                input_ids = mx.array(full)[None, :]
                with mx.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                log_probs = mx.log_softmax(logits, axis=-1)

                # Compute log-likelihood only on y positions
                target_tokens = input_ids[:, 1:]
                log_probs = log_probs[:, :-1, :]
                seq_len = target_tokens.shape[1]

                y_positions = range(max(0, y_start - 1), seq_len)
                token_log_probs = []
                for i in y_positions:
                    token_id = int(target_tokens[0, i].item())
                    token_log_prob = float(log_probs[0, i, token_id].item())
                    token_log_probs.append(token_log_prob)

                if not token_log_probs:
                    return 10.0
                avg_log_prob = sum(token_log_probs) / len(token_log_probs)
                return float(math.exp(-avg_log_prob))

            # Sliding window for long contexts
            # Strategy: ensure y tokens are fully covered, slide backwards from end
            windows = []
            position = len(full)

            while position > 0:
                start = max(0, position - window_size)
                end = position
                windows.append((start, end))
                position -= stride
                if start == 0:
                    break

            # Reverse to process in forward order
            windows.reverse()

            # Accumulate log-probs for y tokens across windows
            token_log_probs_map = {}  # position -> list of log_probs

            for window_start, window_end in windows:
                window_tokens = full[window_start:window_end]
                input_ids = mx.array(window_tokens)[None, :]

                with mx.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                log_probs = mx.log_softmax(logits, axis=-1)

                target_tokens = input_ids[:, 1:]
                log_probs = log_probs[:, :-1, :]

                # Map window positions to global positions
                for local_i in range(len(window_tokens) - 1):
                    global_pos = window_start + local_i + 1  # +1 for target alignment

                    # Only process y tokens
                    if global_pos >= y_start:
                        token_id = int(target_tokens[0, local_i].item())
                        token_log_prob = float(log_probs[0, local_i, token_id].item())

                        if global_pos not in token_log_probs_map:
                            token_log_probs_map[global_pos] = []
                        token_log_probs_map[global_pos].append(token_log_prob)

            # Average log-probs for overlapping positions
            if not token_log_probs_map:
                return 10.0

            final_log_probs = []
            for pos in sorted(token_log_probs_map.keys()):
                probs = token_log_probs_map[pos]
                # Average multiple estimates for same position
                avg_prob = sum(probs) / len(probs)
                final_log_probs.append(avg_prob)

            avg_log_prob = sum(final_log_probs) / len(final_log_probs)
            return float(math.exp(-avg_log_prob))

        return ppl_fn
    except Exception as e:  # pragma: no cover - import-time issues
        raise RuntimeError(
            f"Failed to initialize MLX evaluator for {model_name}: {e}"
        ) from e


def make_together_dspy_ppl_evaluator(
    model_name: str,
) -> Callable[[str, str, str], float]:
    """Create a conditional PPL(y|x,z) evaluator via dspy.LM (Together backend).

    Requires the underlying LM client to support a loglikelihood(context, continuation)
    method (or equivalent). This implementation assumes dspy.LM forwards to a client
    with such capability. No chunking is performed here.

    Note: Validates loglikelihood support on initialization to fail fast.
    """
    import dspy  # type: ignore

    lm = dspy.LM(model=model_name)

    # Early validation: check if loglikelihood is available
    if not hasattr(lm, "loglikelihood"):
        raise RuntimeError(
            f"Together backend via dspy.LM does not expose loglikelihood method.\n"
            f"Model '{model_name}' cannot compute conditional PPL.\n"
            f"Please either:\n"
            f"  1. Switch to MLX backend (--backend mlx) for local evaluation\n"
            f"  2. Use a Together model that supports logprobs/loglikelihood\n"
            f"  3. Enable logprobs in your Together API configuration"
        )

    def ppl_fn(x: str, y: str, z: str) -> float:
        if not y:
            return 10.0
        context = (x or "").strip()
        if z:
            context += "\n\n" + z.strip()

        try:
            res = lm.loglikelihood(context=context, continuation=y)
            # Accept common return shapes
            if isinstance(res, dict) and "loglikelihood" in res:
                ll = float(res["loglikelihood"])  # total log-likelihood
                tokens = max(1, len(y.split()))
                avg_log_prob = ll / tokens
                return float(math.exp(-avg_log_prob))
            if isinstance(res, (int, float)):
                ll = float(res)
                tokens = max(1, len(y.split()))
                avg_log_prob = ll / tokens
                return float(math.exp(-avg_log_prob))
            raise RuntimeError(
                f"Unexpected loglikelihood return format from dspy.LM: {type(res)}"
            )
        except AttributeError as e:
            # Runtime check in case attribute disappears after init
            raise RuntimeError(
                f"Together backend lost loglikelihood capability during execution.\n"
                f"Error: {e}\n"
                f"Please switch to MLX backend for reliable evaluation."
            ) from e

    return ppl_fn


def select_ppl_evaluator(
    backend: str, model_name: str
) -> Callable[[str, str, str], float]:
    """Choose evaluator backend: 'mlx' or 'together'."""
    backend = backend.lower()
    if backend == "mlx":
        return make_mlx_ppl_evaluator(model_name)
    if backend == "together":
        return make_together_dspy_ppl_evaluator(model_name)
    raise ValueError("backend must be one of: 'mlx', 'together'")
