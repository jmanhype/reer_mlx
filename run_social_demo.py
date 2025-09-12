#!/usr/bin/env python3
"""
Demo: Generate social media content using REER + DSPy
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from dspy_program.pipeline import PipelineConfig, SocialContentPipeline


def main():
    print("=" * 60)
    print("🚀 REER × DSPy × MLX Social Content Generator")
    print("=" * 60)

    # Initialize the pipeline with MLX backend
    config = PipelineConfig(
        provider="mlx",
        model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_retries=2,
        temperature=0.7,
    )

    try:
        print("\n📝 Initializing social content pipeline...")
        pipeline = SocialContentPipeline(config)

        # Define content request
        topic = "AI and machine learning trends"
        audience = "tech professionals"
        tone = "engaging and informative"

        print(f"\n🎯 Topic: {topic}")
        print(f"👥 Audience: {audience}")
        print(f"🎨 Tone: {tone}")

        print("\n⚡ Generating social content...")

        # Generate content
        result = pipeline.generate_content(
            topic=topic, audience=audience, tone=tone, platform="twitter"
        )

        if result:
            print("\n✅ Content generated successfully!")
            print("\n" + "=" * 50)
            print("📱 TWITTER POST:")
            print("=" * 50)
            print(result.get("content", "No content generated"))

            if "thread" in result:
                print("\n🧵 THREAD:")
                for i, part in enumerate(result["thread"], 1):
                    print(f"\n[{i}/{len(result['thread'])}]")
                    print(part)

            if "hashtags" in result:
                print(f"\n#️⃣ Hashtags: {' '.join(result['hashtags'])}")

            if "strategy" in result:
                print(f"\n📊 Strategy: {result['strategy']}")
        else:
            print("\n❌ Failed to generate content")

    except ImportError as e:
        print(f"\n⚠️ Missing dependency: {e}")
        print("\n💡 Creating fallback demo...")

        # Fallback: Show what REER would do for social content
        print("\n" + "=" * 50)
        print("REER Social Content Generation Process:")
        print("=" * 50)
        print(
            f"""
1. ANALYZE successful posts in your niche
   - Extract engagement patterns
   - Identify viral elements
   
2. REVERSE-ENGINEER the reasoning:
   - Why did these posts work?
   - What structure did they follow?
   - What emotional triggers were used?
   
3. SYNTHESIZE optimal trajectory:
   - Hook → Context → Value → Engagement
   - Platform-specific optimizations
   - Timing and hashtag strategies
   
4. GENERATE new content following discovered patterns
   
Example output for "{topic}":
        """
        )

        print(
            """
🔥 "The future of AI isn't just about bigger models—it's about 
smarter integration. Here's what tech leaders need to know about 
the 3 trends reshaping our industry in 2024:

1/ Specialized models over generalist ones
2/ Edge computing + AI = game changer  
3/ The rise of AI orchestration layers

Which trend are you most excited about? 🚀

#AI #MachineLearning #TechTrends #Innovation"
        """
        )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
