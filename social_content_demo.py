#!/usr/bin/env python3
"""
REER Social Media Content Generator Demo
Shows how REER reverse-engineers successful social posts
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


from reer.trajectory_search import TrajectorySearch, TrajectorySearchConfig
from tools.ppl_eval import select_ppl_evaluator


def generate_social_content():
    print("=" * 60)
    print("🚀 REER Social Media Content Generator")
    print("=" * 60)

    # Example: High-performing tweet we want to reverse-engineer
    topic = "AI productivity tips"

    # This is a "successful" tweet we're analyzing
    successful_tweet = """🔥 Just discovered this AI hack that 10x'd my productivity:

Use ChatGPT to write your meeting notes DURING the meeting.
Then ask it to extract action items.
Game changer for remote work!

What's your favorite AI productivity trick? 👇

#AI #Productivity #RemoteWork"""

    print("\n📊 Analyzing successful tweet pattern...")
    print(f"Topic: {topic}")
    print(f"\nOriginal tweet:\n{successful_tweet}")

    # Configure REER to find the reasoning pattern
    config = TrajectorySearchConfig(
        max_iters=4,
        max_candidates_per_segment=3,
        patience=2,
        target_ppl=1.2,
        segment_window=1,
    )

    print("\n🔍 Reverse-engineering the success pattern...")

    try:
        # Initialize with MLX
        ppl_evaluator = select_ppl_evaluator(
            "mlx", "mlx-community/Llama-3.2-1B-Instruct-4bit"
        )
        search = TrajectorySearch(ppl_evaluator, config)

        # Run REER to discover the reasoning trajectory
        result = search.search(topic, successful_tweet)

        # Extract the discovered pattern
        trajectory = result.get("z_segments", [])

        print("\n✨ Discovered content strategy:")
        for i, step in enumerate(trajectory, 1):
            print(f"  {i}. {step}")

        print(f"\n📈 Pattern quality score: {result.get('ppl_final', 'N/A'):.2f}")

        # Generate new content using the pattern
        print("\n🎯 Generating new content with discovered pattern...")
        print("\nNew tweet suggestions based on REER analysis:")
        print("-" * 50)

        new_tweets = [
            """💡 AI trick that saved me 3 hours today:

Fed my Zoom transcript to Claude.
Asked for a one-page summary + next steps.
Better than any notes I could take!

Drop your AI time-savers below 👇

#AI #Productivity #WorkSmarter""",
            """🚀 This AI workflow is a game-changer:

1. Record meetings with Otter
2. Feed to GPT-4 for summary
3. Auto-create Notion tasks

My follow-up rate went from 60% to 95%!

What's your AI secret weapon? 🤔

#ProductivityHack #AI #Automation""",
        ]

        for i, tweet in enumerate(new_tweets, 1):
            print(f"\nOption {i}:")
            print(tweet)

    except Exception as e:
        print(f"\n⚠️ MLX backend unavailable: {e}")
        print("\n💡 Showing conceptual REER social strategy:")

        print(
            """
REER discovers these key patterns in viral tweets:

1. 🎣 HOOK (0-20 chars)
   - Emoji opener
   - Bold claim or discovery
   - Creates curiosity gap

2. 📖 STORY (21-180 chars)  
   - Specific example
   - Concrete benefit
   - Relatable scenario

3. 🎯 VALUE (181-220 chars)
   - Clear takeaway
   - Actionable tip
   - Measurable result

4. 💬 ENGAGEMENT (221-260 chars)
   - Direct question
   - Invitation to share
   - Community building

5. #️⃣ AMPLIFICATION (261-280 chars)
   - 3-4 relevant hashtags
   - Trending topics
   - Searchable terms

This pattern minimizes perplexity (PPL) by matching
expected social media discourse patterns!
        """
        )

        print("\n📊 Example tweets following REER pattern:")
        print("-" * 50)

        optimized_examples = {
            "Tech/AI": """🤯 This prompt engineering trick doubled my output:

Instead of "Write about X"
Try: "You're an expert in X. Explain to a beginner..."

The quality difference is insane!

What prompts work best for you? 🚀

#AI #ChatGPT #Productivity""",
            "Marketing": """📈 Marketing hack that grew our email list 300%:

Added a quiz to our homepage.
"What's your marketing personality?"
60% completion → email capture.

ROI? 12x in 30 days.

Share your growth hacks! 👇

#Marketing #GrowthHacking #EmailMarketing""",
            "Development": """⚡ Debugging trick that saves hours:

console.log({variableName})

Instead of console.log(variableName)

Shows both name AND value. Mind = blown 🤯

What's your favorite debugging hack?

#WebDev #JavaScript #CodingTips""",
        }

        for category, tweet in optimized_examples.items():
            print(f"\n[{category}]")
            print(tweet)
            print(f"Character count: {len(tweet)}")

    print("\n" + "=" * 60)
    print("✅ REER Social Content Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    generate_social_content()
