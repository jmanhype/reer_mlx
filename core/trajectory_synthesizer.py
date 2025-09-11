"""T015: Trajectory synthesizer for strategy extraction implementation.

Analyzes sequences of social media posts to extract strategic patterns
and synthesize effective posting trajectories. Uses pattern recognition,
temporal analysis, and strategy clustering to identify successful approaches.
"""

import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import json

from .exceptions import TrajectoryError, StrategyError, ValidationError
from .trace_store import REERTraceStore


@dataclass
class StrategyPattern:
    """Represents an extracted strategy pattern."""

    pattern_id: str
    name: str
    description: str
    features: List[str]
    confidence: float
    frequency: int
    avg_performance: float
    temporal_signatures: Dict[str, Any] = field(default_factory=dict)
    content_patterns: Dict[str, Any] = field(default_factory=dict)
    engagement_patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PostTrajectory:
    """Represents a trajectory of related posts."""

    trajectory_id: str
    posts: List[Dict[str, Any]]
    timeline: List[datetime]
    strategy_evolution: List[str]
    performance_curve: List[float]
    total_performance: float
    trajectory_type: str  # single, thread, campaign, series
    extracted_patterns: List[StrategyPattern] = field(default_factory=list)


@dataclass
class StrategySynthesis:
    """Result of strategy synthesis process."""

    synthesis_id: str
    input_trajectories: List[str]
    extracted_patterns: List[StrategyPattern]
    strategy_recommendations: List[Dict[str, Any]]
    performance_insights: Dict[str, Any]
    temporal_insights: Dict[str, Any]
    content_insights: Dict[str, Any]
    synthesis_confidence: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureExtractor:
    """Extracts various features from post content and metadata."""

    def extract_content_features(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content-based features from a post."""
        text = post.get("text", "")

        features = {
            # Basic metrics
            "character_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": text.count(".") + text.count("!") + text.count("?"),
            # Content patterns
            "hashtag_count": text.count("#"),
            "mention_count": text.count("@"),
            "url_count": len(
                [w for w in text.split() if w.startswith(("http", "www"))]
            ),
            "emoji_count": len([c for c in text if ord(c) > 127]),
            "question_count": text.count("?"),
            "exclamation_count": text.count("!"),
            # Linguistic features
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "punctuation_density": sum(
                1 for c in text if not c.isalnum() and not c.isspace()
            )
            / max(len(text), 1),
            # Content type indicators
            "has_call_to_action": any(
                phrase in text.lower()
                for phrase in [
                    "click",
                    "check",
                    "try",
                    "get",
                    "download",
                    "visit",
                    "learn more",
                    "sign up",
                ]
            ),
            "has_question": "?" in text,
            "has_link": any(word.startswith(("http", "www")) for word in text.split()),
            "has_hashtags": "#" in text,
            "has_mentions": "@" in text,
            "has_numbers": any(c.isdigit() for c in text),
        }

        # Extract hashtags and mentions
        words = text.split()
        features["hashtags"] = [w for w in words if w.startswith("#")]
        features["mentions"] = [w for w in words if w.startswith("@")]

        return features

    def extract_temporal_features(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features from post timing."""
        try:
            timestamp = datetime.fromisoformat(
                post.get("timestamp", "").replace("Z", "+00:00")
            )
        except ValueError:
            # Return default features if timestamp parsing fails
            return {
                "hour_of_day": 12,
                "day_of_week": 1,
                "is_weekend": False,
                "is_business_hours": True,
                "time_zone_offset": 0,
            }

        return {
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "is_weekend": timestamp.weekday() >= 5,
            "is_business_hours": 9 <= timestamp.hour <= 17,
            "time_zone_offset": (
                timestamp.utcoffset().total_seconds() / 3600
                if timestamp.utcoffset()
                else 0
            ),
            "month": timestamp.month,
            "day_of_month": timestamp.day,
            "quarter": (timestamp.month - 1) // 3 + 1,
        }

    def extract_performance_features(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance-related features."""
        metrics = post.get("metrics", {})

        impressions = metrics.get("impressions", 0)
        engagement_rate = metrics.get("engagement_rate", 0)
        retweets = metrics.get("retweets", 0)
        likes = metrics.get("likes", 0)
        replies = metrics.get("replies", 0)

        total_engagement = retweets + likes + replies

        return {
            "impressions": impressions,
            "engagement_rate": engagement_rate,
            "total_engagement": total_engagement,
            "engagement_per_impression": total_engagement / max(impressions, 1),
            "viral_coefficient": retweets / max(total_engagement, 1),
            "like_to_retweet_ratio": likes / max(retweets, 1),
            "reply_engagement_ratio": replies / max(total_engagement, 1),
            "performance_tier": self._classify_performance_tier(engagement_rate),
        }

    def _classify_performance_tier(self, engagement_rate: float) -> str:
        """Classify performance into tiers."""
        if engagement_rate >= 10.0:
            return "exceptional"
        elif engagement_rate >= 5.0:
            return "high"
        elif engagement_rate >= 2.0:
            return "medium"
        elif engagement_rate >= 0.5:
            return "low"
        else:
            return "poor"


class PatternMiner:
    """Mines patterns from post features and trajectories."""

    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def mine_content_patterns(
        self, posts: List[Dict[str, Any]]
    ) -> List[StrategyPattern]:
        """Mine content-based strategy patterns."""
        if not posts:
            return []

        feature_extractor = FeatureExtractor()
        patterns = []

        # Extract features for all posts
        post_features = [
            feature_extractor.extract_content_features(post) for post in posts
        ]
        post_performance = [post.get("score", 0.0) for post in posts]

        # Find high-performing feature combinations
        high_performers = [i for i, score in enumerate(post_performance) if score > 0.7]

        if len(high_performers) < 2:
            return patterns

        # Analyze common features in high-performing posts
        common_features = self._find_common_features(
            [post_features[i] for i in high_performers]
        )

        if common_features:
            avg_performance = np.mean([post_performance[i] for i in high_performers])

            pattern = StrategyPattern(
                pattern_id=f"content_pattern_{len(patterns)}",
                name="High-Performance Content Pattern",
                description=f"Content pattern found in {len(high_performers)} high-performing posts",
                features=list(common_features.keys()),
                confidence=min(common_features.values()),
                frequency=len(high_performers),
                avg_performance=avg_performance,
                content_patterns=common_features,
            )
            patterns.append(pattern)

        return patterns

    def mine_temporal_patterns(
        self, posts: List[Dict[str, Any]]
    ) -> List[StrategyPattern]:
        """Mine temporal strategy patterns."""
        if not posts:
            return []

        feature_extractor = FeatureExtractor()
        patterns = []

        # Group posts by temporal features
        temporal_groups = defaultdict(list)

        for post in posts:
            temporal_features = feature_extractor.extract_temporal_features(post)

            # Group by hour of day
            hour = temporal_features["hour_of_day"]
            temporal_groups[f"hour_{hour}"].append(post)

            # Group by day of week
            day = temporal_features["day_of_week"]
            temporal_groups[f"day_{day}"].append(post)

            # Group by business hours vs off-hours
            if temporal_features["is_business_hours"]:
                temporal_groups["business_hours"].append(post)
            else:
                temporal_groups["off_hours"].append(post)

        # Find temporal patterns with good performance
        for group_name, group_posts in temporal_groups.items():
            if len(group_posts) >= 3:  # Minimum sample size
                avg_score = np.mean([post.get("score", 0.0) for post in group_posts])

                if avg_score > 0.6:  # Good performance threshold
                    pattern = StrategyPattern(
                        pattern_id=f"temporal_pattern_{group_name}",
                        name=f"Temporal Pattern: {group_name}",
                        description=f"Effective posting pattern for {group_name}",
                        features=[group_name],
                        confidence=min(avg_score, 1.0),
                        frequency=len(group_posts),
                        avg_performance=avg_score,
                        temporal_signatures={group_name: avg_score},
                    )
                    patterns.append(pattern)

        return patterns

    def mine_sequence_patterns(
        self, trajectory: PostTrajectory
    ) -> List[StrategyPattern]:
        """Mine sequential patterns within a trajectory."""
        if len(trajectory.posts) < 2:
            return []

        patterns = []

        # Analyze performance trends
        performance_trend = self._analyze_performance_trend(
            trajectory.performance_curve
        )

        if (
            performance_trend["trend"] == "increasing"
            and performance_trend["strength"] > 0.5
        ):
            pattern = StrategyPattern(
                pattern_id=f"seq_pattern_{trajectory.trajectory_id}",
                name="Performance Escalation Pattern",
                description="Sequential posting pattern that builds performance over time",
                features=["sequential_posting", "performance_escalation"],
                confidence=performance_trend["strength"],
                frequency=len(trajectory.posts),
                avg_performance=trajectory.total_performance,
                engagement_patterns={
                    "trend": performance_trend["trend"],
                    "escalation_rate": performance_trend.get("slope", 0),
                },
            )
            patterns.append(pattern)

        return patterns

    def _find_common_features(
        self, feature_sets: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Find features that appear frequently across feature sets."""
        if not feature_sets:
            return {}

        feature_counts = defaultdict(int)
        binary_features = set()

        # Count feature occurrences
        for features in feature_sets:
            for feature, value in features.items():
                if isinstance(value, bool) and value:
                    feature_counts[feature] += 1
                    binary_features.add(feature)
                elif isinstance(value, (int, float)) and value > 0:
                    feature_counts[feature] += 1

        # Calculate support (frequency)
        total_sets = len(feature_sets)
        common_features = {}

        for feature, count in feature_counts.items():
            support = count / total_sets
            if support >= self.min_support:
                common_features[feature] = support

        return common_features

    def _analyze_performance_trend(
        self, performance_curve: List[float]
    ) -> Dict[str, Any]:
        """Analyze the trend in performance over time."""
        if len(performance_curve) < 2:
            return {"trend": "flat", "strength": 0.0}

        # Calculate linear trend
        x = np.arange(len(performance_curve))
        y = np.array(performance_curve)

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)

        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0

        # Determine trend direction
        if slope > 0.05:
            trend = "increasing"
        elif slope < -0.05:
            trend = "decreasing"
        else:
            trend = "flat"

        return {
            "trend": trend,
            "slope": slope,
            "strength": abs(correlation),
            "correlation": correlation,
        }


class TrajectoryBuilder:
    """Builds post trajectories from trace data."""

    def __init__(self, max_time_gap: timedelta = timedelta(hours=24)):
        self.max_time_gap = max_time_gap

    def build_trajectories(self, traces: List[Dict[str, Any]]) -> List[PostTrajectory]:
        """Build trajectories from a list of traces."""
        if not traces:
            return []

        # Sort traces by timestamp
        sorted_traces = sorted(
            traces, key=lambda t: self._parse_timestamp(t.get("timestamp", ""))
        )

        trajectories = []

        # Group by source_post_id for thread detection
        post_groups = defaultdict(list)
        for trace in sorted_traces:
            source_id = trace.get("source_post_id", "")
            post_groups[source_id].append(trace)

        # Build trajectories
        for source_id, group_traces in post_groups.items():
            if len(group_traces) == 1:
                # Single post trajectory
                trajectory = self._build_single_trajectory(group_traces[0])
                trajectories.append(trajectory)
            else:
                # Multi-post trajectory (thread or series)
                trajectory = self._build_multi_trajectory(group_traces)
                trajectories.append(trajectory)

        return trajectories

    def _build_single_trajectory(self, trace: Dict[str, Any]) -> PostTrajectory:
        """Build trajectory for a single post."""
        timestamp = self._parse_timestamp(trace.get("timestamp", ""))

        return PostTrajectory(
            trajectory_id=f"single_{trace.get('id', 'unknown')}",
            posts=[trace],
            timeline=[timestamp],
            strategy_evolution=trace.get("strategy_features", []),
            performance_curve=[trace.get("score", 0.0)],
            total_performance=trace.get("score", 0.0),
            trajectory_type="single",
        )

    def _build_multi_trajectory(self, traces: List[Dict[str, Any]]) -> PostTrajectory:
        """Build trajectory for multiple related posts."""
        # Sort by timestamp
        sorted_traces = sorted(
            traces, key=lambda t: self._parse_timestamp(t.get("timestamp", ""))
        )

        timeline = [
            self._parse_timestamp(t.get("timestamp", "")) for t in sorted_traces
        ]
        performance_curve = [t.get("score", 0.0) for t in sorted_traces]

        # Combine strategy features
        all_features = []
        for trace in sorted_traces:
            all_features.extend(trace.get("strategy_features", []))

        # Determine trajectory type
        max_gap = (
            max(
                (timeline[i + 1] - timeline[i]).total_seconds() / 3600
                for i in range(len(timeline) - 1)
            )
            if len(timeline) > 1
            else 0
        )

        trajectory_type = "thread" if max_gap <= 1 else "series"

        return PostTrajectory(
            trajectory_id=f"{trajectory_type}_{sorted_traces[0].get('id', 'unknown')}",
            posts=sorted_traces,
            timeline=timeline,
            strategy_evolution=list(set(all_features)),  # Unique features
            performance_curve=performance_curve,
            total_performance=np.mean(performance_curve),
            trajectory_type=trajectory_type,
        )

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime."""
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)


class REERTrajectorySynthesizer:
    """Main trajectory synthesizer for strategy extraction.

    Analyzes historical posting data to extract effective strategy patterns
    and synthesize actionable insights for future posting strategies.
    """

    def __init__(
        self,
        trace_store: Optional[REERTraceStore] = None,
        min_pattern_support: float = 0.1,
        min_pattern_confidence: float = 0.6,
    ):
        """Initialize trajectory synthesizer.

        Args:
            trace_store: Optional trace store for data access
            min_pattern_support: Minimum support for pattern mining
            min_pattern_confidence: Minimum confidence for patterns
        """
        self.trace_store = trace_store
        self.pattern_miner = PatternMiner(min_pattern_support, min_pattern_confidence)
        self.trajectory_builder = TrajectoryBuilder()
        self.feature_extractor = FeatureExtractor()

    async def synthesize_strategies(
        self,
        traces: Optional[List[Dict[str, Any]]] = None,
        provider_filter: Optional[str] = None,
        min_score: float = 0.0,
        analysis_window_days: int = 30,
    ) -> StrategySynthesis:
        """Synthesize strategies from trace data.

        Args:
            traces: Optional list of traces (if None, loads from store)
            provider_filter: Filter traces by provider
            min_score: Minimum score threshold for analysis
            analysis_window_days: Analysis window in days

        Returns:
            StrategySynthesis with extracted patterns and recommendations
        """
        try:
            # Load traces if not provided
            if traces is None:
                if not self.trace_store:
                    raise TrajectoryError(
                        "No traces provided and no trace store configured"
                    )

                since = datetime.now(timezone.utc) - timedelta(
                    days=analysis_window_days
                )
                traces = await self.trace_store.query_traces(
                    provider=provider_filter, min_score=min_score, since=since
                )

            if not traces:
                raise TrajectoryError("No traces available for synthesis")

            # Build trajectories
            trajectories = self.trajectory_builder.build_trajectories(traces)

            # Extract patterns
            all_patterns = []

            # Mine content patterns
            content_patterns = self.pattern_miner.mine_content_patterns(traces)
            all_patterns.extend(content_patterns)

            # Mine temporal patterns
            temporal_patterns = self.pattern_miner.mine_temporal_patterns(traces)
            all_patterns.extend(temporal_patterns)

            # Mine sequence patterns from trajectories
            for trajectory in trajectories:
                sequence_patterns = self.pattern_miner.mine_sequence_patterns(
                    trajectory
                )
                all_patterns.extend(sequence_patterns)

            # Generate insights and recommendations
            performance_insights = self._analyze_performance_insights(
                traces, trajectories
            )
            temporal_insights = self._analyze_temporal_insights(traces)
            content_insights = self._analyze_content_insights(traces)

            strategy_recommendations = self._generate_recommendations(
                all_patterns, performance_insights, temporal_insights, content_insights
            )

            # Calculate synthesis confidence
            synthesis_confidence = self._calculate_synthesis_confidence(
                all_patterns, traces, trajectories
            )

            return StrategySynthesis(
                synthesis_id=f"synthesis_{int(datetime.now().timestamp())}",
                input_trajectories=[t.trajectory_id for t in trajectories],
                extracted_patterns=all_patterns,
                strategy_recommendations=strategy_recommendations,
                performance_insights=performance_insights,
                temporal_insights=temporal_insights,
                content_insights=content_insights,
                synthesis_confidence=synthesis_confidence,
            )

        except Exception as e:
            raise TrajectoryError(
                f"Strategy synthesis failed: {str(e)}",
                details={"trace_count": len(traces) if traces else 0},
                original_error=e,
            )

    def _analyze_performance_insights(
        self, traces: List[Dict[str, Any]], trajectories: List[PostTrajectory]
    ) -> Dict[str, Any]:
        """Analyze performance insights from traces and trajectories."""
        scores = [trace.get("score", 0.0) for trace in traces]

        return {
            "total_posts_analyzed": len(traces),
            "avg_performance": np.mean(scores),
            "performance_std": np.std(scores),
            "top_performance": np.max(scores) if scores else 0.0,
            "performance_distribution": {
                "excellent": len([s for s in scores if s >= 0.8]),
                "good": len([s for s in scores if 0.6 <= s < 0.8]),
                "average": len([s for s in scores if 0.4 <= s < 0.6]),
                "poor": len([s for s in scores if s < 0.4]),
            },
            "trajectory_types": Counter(t.trajectory_type for t in trajectories),
            "avg_trajectory_performance": {
                ttype: np.mean(
                    [
                        t.total_performance
                        for t in trajectories
                        if t.trajectory_type == ttype
                    ]
                )
                for ttype in set(t.trajectory_type for t in trajectories)
            },
        }

    def _analyze_temporal_insights(
        self, traces: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in posting behavior."""
        temporal_data = []

        for trace in traces:
            temporal_features = self.feature_extractor.extract_temporal_features(trace)
            temporal_features["score"] = trace.get("score", 0.0)
            temporal_data.append(temporal_features)

        if not temporal_data:
            return {}

        # Analyze performance by hour
        hour_performance = defaultdict(list)
        for data in temporal_data:
            hour_performance[data["hour_of_day"]].append(data["score"])

        best_hours = sorted(
            [(hour, np.mean(scores)) for hour, scores in hour_performance.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        # Analyze performance by day of week
        day_performance = defaultdict(list)
        for data in temporal_data:
            day_performance[data["day_of_week"]].append(data["score"])

        best_days = sorted(
            [(day, np.mean(scores)) for day, scores in day_performance.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        return {
            "best_posting_hours": [{"hour": h, "avg_score": s} for h, s in best_hours],
            "best_posting_days": [{"day": d, "avg_score": s} for d, s in best_days],
            "business_hours_performance": (
                np.mean([d["score"] for d in temporal_data if d["is_business_hours"]])
                if temporal_data
                else 0.0
            ),
            "weekend_performance": (
                np.mean([d["score"] for d in temporal_data if d["is_weekend"]])
                if temporal_data
                else 0.0
            ),
        }

    def _analyze_content_insights(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content patterns and their performance."""
        content_data = []

        for trace in traces:
            # Extract content from seed_params or try to reconstruct
            seed_params = trace.get("seed_params", {})
            content_info = {
                "topic": seed_params.get("topic", ""),
                "style": seed_params.get("style", ""),
                "length": seed_params.get("length", 0),
                "thread_size": seed_params.get("thread_size", 1),
                "strategy_features": trace.get("strategy_features", []),
                "score": trace.get("score", 0.0),
            }
            content_data.append(content_info)

        if not content_data:
            return {}

        # Analyze feature performance
        feature_performance = defaultdict(list)
        for data in content_data:
            for feature in data["strategy_features"]:
                feature_performance[feature].append(data["score"])

        top_features = sorted(
            [
                (feature, np.mean(scores))
                for feature, scores in feature_performance.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Analyze length performance
        length_groups = {
            "short": [d["score"] for d in content_data if d["length"] <= 100],
            "medium": [d["score"] for d in content_data if 100 < d["length"] <= 200],
            "long": [d["score"] for d in content_data if d["length"] > 200],
        }

        return {
            "top_performing_features": [
                {"feature": f, "avg_score": s} for f, s in top_features
            ],
            "optimal_length_ranges": {
                length_type: {"avg_score": np.mean(scores), "count": len(scores)}
                for length_type, scores in length_groups.items()
                if scores
            },
            "most_common_styles": Counter(d["style"] for d in content_data).most_common(
                5
            ),
            "thread_vs_single_performance": {
                "single_posts": np.mean(
                    [d["score"] for d in content_data if d["thread_size"] == 1]
                ),
                "thread_posts": np.mean(
                    [d["score"] for d in content_data if d["thread_size"] > 1]
                ),
            },
        }

    def _generate_recommendations(
        self,
        patterns: List[StrategyPattern],
        performance_insights: Dict[str, Any],
        temporal_insights: Dict[str, Any],
        content_insights: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate actionable strategy recommendations."""
        recommendations = []

        # Timing recommendations
        if temporal_insights.get("best_posting_hours"):
            best_hour = temporal_insights["best_posting_hours"][0]
            recommendations.append(
                {
                    "type": "timing",
                    "priority": "high",
                    "title": "Optimal Posting Time",
                    "description": f"Post around {best_hour['hour']}:00 for best performance",
                    "expected_improvement": f"{best_hour['avg_score']:.1%}",
                    "confidence": 0.8,
                }
            )

        # Content feature recommendations
        if content_insights.get("top_performing_features"):
            top_feature = content_insights["top_performing_features"][0]
            recommendations.append(
                {
                    "type": "content",
                    "priority": "high",
                    "title": "High-Impact Content Feature",
                    "description": f"Include '{top_feature['feature']}' in your posts",
                    "expected_improvement": f"{top_feature['avg_score']:.1%}",
                    "confidence": 0.75,
                }
            )

        # Pattern-based recommendations
        high_confidence_patterns = [p for p in patterns if p.confidence > 0.7]
        for pattern in high_confidence_patterns[:3]:  # Top 3 patterns
            recommendations.append(
                {
                    "type": "pattern",
                    "priority": "medium",
                    "title": f"Apply {pattern.name}",
                    "description": pattern.description,
                    "expected_improvement": f"{pattern.avg_performance:.1%}",
                    "confidence": pattern.confidence,
                    "features": pattern.features,
                }
            )

        return recommendations

    def _calculate_synthesis_confidence(
        self,
        patterns: List[StrategyPattern],
        traces: List[Dict[str, Any]],
        trajectories: List[PostTrajectory],
    ) -> float:
        """Calculate confidence in the synthesis results."""
        factors = []

        # Data volume factor
        data_factor = min(len(traces) / 100, 1.0)  # Full confidence at 100+ traces
        factors.append(data_factor)

        # Pattern quality factor
        if patterns:
            avg_pattern_confidence = np.mean([p.confidence for p in patterns])
            factors.append(avg_pattern_confidence)
        else:
            factors.append(0.3)  # Low confidence if no patterns

        # Trajectory diversity factor
        trajectory_types = set(t.trajectory_type for t in trajectories)
        diversity_factor = min(len(trajectory_types) / 3, 1.0)  # Full at 3+ types
        factors.append(diversity_factor)

        # Performance spread factor
        scores = [t.get("score", 0.0) for t in traces]
        if scores:
            score_std = np.std(scores)
            spread_factor = min(score_std * 2, 1.0)  # More spread = more confidence
            factors.append(spread_factor)
        else:
            factors.append(0.5)

        return float(np.mean(factors))

    async def export_synthesis(
        self, synthesis: StrategySynthesis, output_path: Path
    ) -> None:
        """Export synthesis results to JSON file."""
        try:
            # Convert to serializable format
            export_data = {
                "synthesis_id": synthesis.synthesis_id,
                "created_at": synthesis.created_at.isoformat(),
                "input_trajectories": synthesis.input_trajectories,
                "synthesis_confidence": synthesis.synthesis_confidence,
                "extracted_patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "name": p.name,
                        "description": p.description,
                        "features": p.features,
                        "confidence": p.confidence,
                        "frequency": p.frequency,
                        "avg_performance": p.avg_performance,
                        "temporal_signatures": p.temporal_signatures,
                        "content_patterns": p.content_patterns,
                        "engagement_patterns": p.engagement_patterns,
                    }
                    for p in synthesis.extracted_patterns
                ],
                "strategy_recommendations": synthesis.strategy_recommendations,
                "performance_insights": synthesis.performance_insights,
                "temporal_insights": synthesis.temporal_insights,
                "content_insights": synthesis.content_insights,
            }

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        except Exception as e:
            raise TrajectoryError(
                f"Failed to export synthesis: {str(e)}",
                details={"output_path": str(output_path)},
                original_error=e,
            )
