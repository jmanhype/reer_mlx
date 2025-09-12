#!/usr/bin/env python3
"""
Simple REER demo - Shows basic trajectory search functionality
"""

from reer.trajectory_search import TrajectorySearch, TrajectorySearchConfig


def main():
    print("=" * 60)
    print("REER (REverse-Engineered Reasoning) Demo")
    print("=" * 60)

    # Sample input and output
    x = "What is machine learning?"
    y = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."

    print(f"\n📝 Input (x): {x}")
    print(f"📝 Output (y): {y[:80]}...")

    # Configure trajectory search
    config = TrajectorySearchConfig(
        max_iters=3,  # Reduced for demo
        max_candidates_per_segment=2,  # Reduced for demo
        patience=1,
        target_ppl=1.1,
        segment_window=1,
    )

    print("\n⚙️ Configuration:")
    print(f"  - Max iterations: {config.max_iters}")
    print(f"  - Candidates per segment: {config.max_candidates_per_segment}")
    print(f"  - Target PPL: {config.target_ppl}")

    # Create search instance (note: this will fail without proper backend)
    print("\n🔄 Initializing trajectory search...")
    try:
        # Try with a mock backend for demo purposes
        search = TrajectorySearch(
            config=config,
            backend="mock",  # This will fail but shows the structure
            model="demo-model",
        )

        # Run trajectory search
        print("🔍 Running trajectory optimization...")
        result = search.search(x, y)

        print("\n✅ Optimization complete!")
        print("📊 Result:")
        print(f"  - Original PPL: {result.get('original_ppl', 'N/A')}")
        print(f"  - Optimized PPL: {result.get('optimized_ppl', 'N/A')}")
        print(f"  - Reasoning chain: {result.get('z', 'N/A')[:100]}...")

    except Exception as e:
        print("\n⚠️ Note: Backend initialization failed (expected in demo mode)")
        print(f"   Error: {str(e)}")
        print("\n💡 To run actual trajectory search, you need:")
        print("   1. MLX backend with Apple Silicon (M1/M2/M3)")
        print("   2. Or Together API key with compatible model")
        print("\n📚 Example commands:")
        print("   # With MLX (local):")
        print("   python scripts/reer_synthesize.py examples/qa.json \\")
        print("     --backend mlx --model 'mlx-community/model-name'")
        print("\n   # With Together API:")
        print("   export TOGETHER_API_KEY='your-key'")
        print("   python scripts/reer_synthesize.py examples/qa.json \\")
        print("     --backend together --model 'meta-llama/model-name'")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
