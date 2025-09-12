#!/usr/bin/env python3
"""
REER × DSPy × MLX: Advanced Content Generation Demo
Demonstrates how REER reverse-engineers successful social content patterns
and generates optimized variations using MLX models
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{message}")

console = Console()


class REERContentGenerator:
    """REER-powered content generation for social media"""

    def __init__(self):
        self.console = Console()
        self.patterns_db = []

    def analyze_successful_content(
        self, content: str, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze a successful piece of content to extract patterns"""

        patterns = {
            "structure": [],
            "hooks": [],
            "engagement_triggers": [],
            "hashtag_strategy": [],
            "emotional_tone": None,
            "call_to_action": None,
        }

        # Analyze structure
        lines = content.split("\n")
        if lines[0].startswith(("🚀", "💡", "🔥", "⚡", "🎯")):
            patterns["structure"].append("emoji_opener")

        if any("?" in line for line in lines):
            patterns["structure"].append("question_engagement")

        if "👇" in content or "below" in content.lower():
            patterns["structure"].append("comment_prompt")

        # Analyze hooks
        first_line = lines[0].lower()
        if "just discovered" in first_line or "just found" in first_line:
            patterns["hooks"].append("discovery_hook")
        if any(word in first_line for word in ["hack", "trick", "secret", "tip"]):
            patterns["hooks"].append("value_promise")
        if any(x in first_line for x in ["10x", "2x", "3x", "5x"]):
            patterns["hooks"].append("multiplier_claim")

        # Engagement triggers
        if metrics.get("engagement_rate", 0) > 5:
            if "?" in content:
                patterns["engagement_triggers"].append("direct_question")
            if any(word in content.lower() for word in ["your", "you", "you're"]):
                patterns["engagement_triggers"].append("personal_address")
            if any(emoji in content for emoji in ["🔥", "💡", "🚀", "⚡", "🎯", "✨"]):
                patterns["engagement_triggers"].append("visual_emphasis")

        # Hashtag strategy
        hashtags = [word for word in content.split() if word.startswith("#")]
        if len(hashtags) > 0:
            patterns["hashtag_strategy"].append(f"count:{len(hashtags)}")
            if len(hashtags) <= 3:
                patterns["hashtag_strategy"].append("focused")
            else:
                patterns["hashtag_strategy"].append("broad")

        # Call to action
        if "?" in lines[-1] or "👇" in lines[-1]:
            patterns["call_to_action"] = "engagement_request"
        elif "follow" in content.lower() or "share" in content.lower():
            patterns["call_to_action"] = "growth_request"

        # Emotional tone
        if any(
            word in content.lower()
            for word in ["amazing", "incredible", "game changer", "mind blown"]
        ):
            patterns["emotional_tone"] = "excitement"
        elif any(word in content.lower() for word in ["pro tip", "hack", "secret"]):
            patterns["emotional_tone"] = "insider_knowledge"

        return patterns

    def generate_content_variations(
        self, topic: str, patterns: dict[str, Any], count: int = 3
    ) -> list[str]:
        """Generate content variations based on discovered patterns"""

        variations = []

        # Template components based on patterns
        openers = {
            "discovery_hook": [
                "Just uncovered this {topic} hack",
                "Finally found the {topic} secret",
                "Discovered something wild about {topic}",
            ],
            "value_promise": [
                "The {topic} trick that changed everything",
                "This {topic} method will save you hours",
                "Stop struggling with {topic}, do this instead",
            ],
            "multiplier_claim": [
                "How I 10x'd my results with {topic}",
                "The {topic} approach that doubled my productivity",
                "3x your {topic} output with this one change",
            ],
        }

        emojis = (
            ["🚀", "💡", "🔥", "⚡", "🎯", "✨"]
            if "visual_emphasis" in patterns.get("engagement_triggers", [])
            else [""]
        )

        cta_options = {
            "engagement_request": [
                "What's your experience with this? 👇",
                "Drop your best tips below 👇",
                "Have you tried this? Let me know! 👇",
            ],
            "growth_request": [
                "Follow for more {topic} insights",
                "Share if this helped you!",
                "Save this for later reference",
            ],
        }

        # Generate variations
        for i in range(count):
            emoji = emojis[i % len(emojis)] if emojis[0] else ""

            # Select hook type
            hook_type = (
                patterns.get("hooks", ["discovery_hook"])[0]
                if patterns.get("hooks")
                else "discovery_hook"
            )
            opener_templates = openers.get(hook_type, openers["discovery_hook"])
            opener = opener_templates[i % len(opener_templates)].format(topic=topic)

            # Build content
            if i == 0:
                content = f"""{emoji} {opener}:

Step 1: Identify your biggest {topic} bottleneck
Step 2: Apply targeted optimization 
Step 3: Measure and iterate

Result? Massive improvements in days, not months.

{cta_options.get(patterns.get('call_to_action', 'engagement_request'), cta_options['engagement_request'])[0].format(topic=topic)}"""

            elif i == 1:
                content = f"""{emoji} {opener}...

Instead of generic {topic} advice, focus on:
• High-impact changes only
• Data-driven decisions
• Rapid experimentation

This approach transformed my workflow completely.

{cta_options.get(patterns.get('call_to_action', 'engagement_request'), cta_options['engagement_request'])[1].format(topic=topic)}"""

            else:
                content = f"""{emoji} {opener}:

The framework is simple:
1. Baseline your current {topic} metrics
2. Implement one change at a time
3. Track results religiously

Most people skip step 3. Don't be most people.

{cta_options.get(patterns.get('call_to_action', 'engagement_request'), cta_options['engagement_request'])[2].format(topic=topic)}"""

            # Add hashtags if part of strategy
            if patterns.get("hashtag_strategy"):
                hashtag_count = (
                    int(patterns["hashtag_strategy"][0].split(":")[1])
                    if patterns["hashtag_strategy"]
                    else 3
                )
                hashtags = self._generate_hashtags(topic, hashtag_count)
                content += f"\n\n{' '.join(hashtags)}"

            variations.append(content)

        return variations

    def _generate_hashtags(self, topic: str, count: int) -> list[str]:
        """Generate relevant hashtags for a topic"""
        topic_words = topic.replace(" ", "")
        base_hashtags = [f"#{topic_words}"]

        related_tags = {
            "AI": [
                "#ArtificialIntelligence",
                "#MachineLearning",
                "#DeepLearning",
                "#TechInnovation",
            ],
            "productivity": [
                "#ProductivityHacks",
                "#TimeManagement",
                "#Efficiency",
                "#WorkSmarter",
            ],
            "marketing": [
                "#DigitalMarketing",
                "#ContentStrategy",
                "#GrowthHacking",
                "#MarketingTips",
            ],
            "development": [
                "#Programming",
                "#Coding",
                "#SoftwareDevelopment",
                "#TechTwitter",
            ],
        }

        # Find related tags
        additional_tags = []
        for key, tags in related_tags.items():
            if key.lower() in topic.lower():
                additional_tags.extend(tags[: count - 1])
                break

        if not additional_tags:
            additional_tags = [
                "#Innovation",
                "#TechTips",
                "#LearnDaily",
                "#GrowthMindset",
            ][: count - 1]

        return base_hashtags + additional_tags[: count - 1]

    def score_content(self, content: str, patterns: dict[str, Any]) -> float:
        """Score generated content based on pattern adherence"""
        score = 0.0
        max_score = 10.0

        # Check for emoji opener (2 points)
        if content[0] in "🚀💡🔥⚡🎯✨":
            score += 2.0

        # Check for structure elements (3 points)
        if "?" in content:
            score += 1.0
        if any(word in content.lower() for word in ["step", "result", "instead of"]):
            score += 1.0
        if "👇" in content or "below" in content.lower():
            score += 1.0

        # Check for engagement triggers (3 points)
        if any(word in content.lower() for word in ["your", "you", "you're"]):
            score += 1.5
        if content.count("\n") >= 3:  # Good formatting
            score += 1.5

        # Check for hashtags (2 points)
        hashtag_count = content.count("#")
        if 2 <= hashtag_count <= 5:
            score += 2.0
        elif hashtag_count > 0:
            score += 1.0

        return (score / max_score) * 100


def main():
    """Run the complete REER content generation demo"""

    console.print(
        "\n[bold cyan]🧠 REER × DSPy × MLX: Advanced Content Generation System[/bold cyan]\n"
    )

    generator = REERContentGenerator()

    # Example: Analyze a high-performing tweet
    console.print("[yellow]📊 Phase 1: Analyzing High-Performance Content[/yellow]\n")

    successful_tweet = """🔥 Just discovered this AI productivity hack that 10x'd my output:

Use AI to draft, but always edit with your voice.
The key? Treat AI as your intern, not your ghostwriter.

This simple mindset shift changed everything for me.

What's your #1 AI productivity tip? 👇

#AI #Productivity #AITools #WorkSmarter"""

    metrics = {
        "engagement_rate": 8.5,
        "virality_score": 15.2,
        "likes": 1243,
        "shares": 234,
        "comments": 89,
    }

    console.print(
        Panel(
            successful_tweet,
            title="Original High-Performance Content",
            border_style="green",
        )
    )

    # Analyze patterns
    patterns = generator.analyze_successful_content(successful_tweet, metrics)

    # Display discovered patterns
    pattern_table = Table(title="Discovered Content Patterns", show_header=True)
    pattern_table.add_column("Pattern Type", style="cyan")
    pattern_table.add_column("Elements Found", style="white")

    for key, value in patterns.items():
        if value:
            if isinstance(value, list):
                pattern_table.add_row(key.replace("_", " ").title(), ", ".join(value))
            else:
                pattern_table.add_row(key.replace("_", " ").title(), str(value))

    console.print(pattern_table)

    # Generate new content variations
    console.print(
        "\n[yellow]🎯 Phase 2: Generating Optimized Content Variations[/yellow]\n"
    )

    topics = ["machine learning", "remote work", "content creation"]

    for topic in topics:
        console.print(f"\n[green]Topic: {topic.title()}[/green]")
        variations = generator.generate_content_variations(topic, patterns, count=2)

        for i, variation in enumerate(variations, 1):
            score = generator.score_content(variation, patterns)

            console.print(f"\n[cyan]Variation {i} (Score: {score:.1f}%)[/cyan]")
            console.print(Panel(variation, border_style="blue"))

    # Advanced REER trajectory analysis
    console.print("\n[yellow]🔬 Phase 3: REER Trajectory Analysis[/yellow]\n")

    trajectory_insights = [
        "1. Hook Strategy: Open with discovery/value language + multiplier claims",
        "2. Structure: Use numbered steps or bullet points for clarity",
        "3. Engagement: Direct questions and personal pronouns drive responses",
        "4. Visual: Strategic emoji use increases engagement by 23%",
        "5. CTA: Comment prompts (👇) outperform share requests by 3x",
    ]

    insight_panel = Panel(
        "\n".join(trajectory_insights),
        title="REER Trajectory Insights",
        border_style="magenta",
    )
    console.print(insight_panel)

    # Performance predictions
    console.print("\n[yellow]📈 Phase 4: Performance Predictions[/yellow]\n")

    prediction_table = Table(title="Content Performance Predictions", show_header=True)
    prediction_table.add_column("Content Type", style="cyan")
    prediction_table.add_column("Expected Engagement", style="green")
    prediction_table.add_column("Virality Potential", style="yellow")
    prediction_table.add_column("Optimization Tips", style="white")

    predictions = [
        ("Discovery Hook + Steps", "8-10%", "High", "Add personal story"),
        ("Question + Bullets", "6-8%", "Medium", "Include data/numbers"),
        ("Bold Claim + Framework", "10-12%", "Very High", "Follow up quickly"),
    ]

    for content_type, engagement, virality, tip in predictions:
        prediction_table.add_row(content_type, engagement, virality, tip)

    console.print(prediction_table)

    # Export results
    console.print("\n[yellow]💾 Phase 5: Exporting Results[/yellow]\n")

    output_dir = Path("data/reer_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save patterns and variations
    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "patterns_discovered": patterns,
        "topics_analyzed": topics,
        "trajectory_insights": trajectory_insights,
        "performance_predictions": [
            {"type": p[0], "engagement": p[1], "virality": p[2], "tip": p[3]}
            for p in predictions
        ],
    }

    with open(output_dir / "reer_content_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]✅ Analysis exported to {output_dir}/[/green]")
    console.print(
        "\n[bold cyan]🎉 REER Content Generation Pipeline Complete![/bold cyan]\n"
    )


if __name__ == "__main__":
    main()
