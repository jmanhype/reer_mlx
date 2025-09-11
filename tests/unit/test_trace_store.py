"""T041: Comprehensive unit tests for trace_store.py

Tests for append-only JSONL storage, validation, concurrency, and query functionality.
Covers async/sync APIs, file locking, and error handling with mock file I/O.
"""

import asyncio
from datetime import UTC, datetime
import json
from unittest.mock import mock_open, patch
from uuid import UUID, uuid4

from pydantic import ValidationError as PydanticValidationError
import pytest

from core.exceptions import TraceStoreError, ValidationError
from core.trace_store import (
    REERTraceStore,
    TraceRecord,
)


class TestTraceRecord:
    """Test TraceRecord Pydantic model validation."""

    def test_valid_trace_record(self):
        """Test creating a valid trace record."""
        trace_data = {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags", "question_pattern"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

        record = TraceRecord(**trace_data)
        assert record.id == trace_data["id"]
        assert record.score == 0.85
        assert record.strategy_features == ["hashtags", "question_pattern"]

    def test_invalid_uuid(self):
        """Test validation fails with invalid UUID."""
        trace_data = {
            "id": "invalid-uuid",
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

        with pytest.raises(PydanticValidationError) as exc_info:
            TraceRecord(**trace_data)
        assert "id must be a valid UUID v4" in str(exc_info.value)

    def test_invalid_timestamp(self):
        """Test validation fails with invalid timestamp."""
        trace_data = {
            "id": str(uuid4()),
            "timestamp": "not-a-timestamp",
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

        with pytest.raises(PydanticValidationError) as exc_info:
            TraceRecord(**trace_data)
        assert "timestamp must be ISO 8601 format" in str(exc_info.value)

    def test_invalid_score_range(self):
        """Test validation fails with score outside 0-1 range."""
        trace_data = {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 1.5,  # Invalid: > 1.0
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

        with pytest.raises(PydanticValidationError):
            TraceRecord(**trace_data)

    def test_invalid_seed_params_missing_fields(self):
        """Test validation fails with missing seed_params fields."""
        trace_data = {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                # Missing style, length, thread_size
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

        with pytest.raises(PydanticValidationError) as exc_info:
            TraceRecord(**trace_data)
        assert "seed_params must contain" in str(exc_info.value)

    def test_invalid_metrics_negative_values(self):
        """Test validation fails with negative metric values."""
        trace_data = {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 0.85,
            "metrics": {
                "impressions": -100,  # Invalid: negative
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

        with pytest.raises(PydanticValidationError) as exc_info:
            TraceRecord(**trace_data)
        assert "must be non-negative integer" in str(exc_info.value)

    def test_invalid_provider_pattern(self):
        """Test validation fails with invalid provider pattern."""
        trace_data = {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags"],
            "provider": "invalid-provider-format",  # Missing :: pattern
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

        with pytest.raises(PydanticValidationError):
            TraceRecord(**trace_data)

    def test_empty_strategy_features(self):
        """Test validation fails with empty strategy features."""
        trace_data = {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": [],  # Invalid: empty list
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

        with pytest.raises(PydanticValidationError):
            TraceRecord(**trace_data)


class TestREERTraceStore:
    """Test REERTraceStore functionality."""

    @pytest.fixture
    def temp_trace_file(self, tmp_path):
        """Create a temporary trace file."""
        return tmp_path / "test_traces.jsonl"

    @pytest.fixture
    def sample_trace(self):
        """Create a sample trace for testing."""
        return {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags", "question_pattern"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

    @pytest.fixture
    def trace_store(self, temp_trace_file):
        """Create a trace store instance."""
        return REERTraceStore(temp_trace_file, validate_on_write=True)

    def test_initialization(self, temp_trace_file):
        """Test trace store initialization."""
        store = REERTraceStore(temp_trace_file)
        assert store.file_path == temp_trace_file
        assert store.validate_on_write is True
        assert store._lock is not None
        assert temp_trace_file.parent.exists()

    def test_initialization_with_schema(self, temp_trace_file, tmp_path):
        """Test trace store initialization with schema file."""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type": "object"}')

        store = REERTraceStore(temp_trace_file, schema_path=schema_file)
        assert store.schema_path == schema_file

    @pytest.mark.asyncio
    async def test_append_trace_valid(self, trace_store, sample_trace):
        """Test appending a valid trace."""
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("fcntl.flock"):
                trace_id = await trace_store.append_trace(sample_trace)

                assert trace_id == sample_trace["id"]
                mock_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_append_trace_generates_id_and_timestamp(self, trace_store):
        """Test that missing ID and timestamp are generated."""
        trace_without_id = {
            "source_post_id": "post_123",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "llm_based",
                "confidence": 0.92,
            },
        }

        # Mock the validation to avoid Pydantic errors during the test
        with patch.object(trace_store, "_validate_trace_async") as mock_validate:
            mock_validate.return_value = None  # No validation errors

            with patch("builtins.open", mock_open()), patch("fcntl.flock"):
                trace_id = await trace_store.append_trace(trace_without_id)

                assert trace_id is not None
                # Verify UUID format
                UUID(trace_id, version=4)
                # Verify timestamp was added
                assert "timestamp" in trace_without_id
                assert "id" in trace_without_id

    @pytest.mark.asyncio
    async def test_append_trace_invalid_data(self, trace_store):
        """Test appending invalid trace data."""
        invalid_trace = {
            "id": "invalid-uuid",
            "source_post_id": "post_123",
            # Missing required fields
        }

        with pytest.raises(ValidationError):
            await trace_store.append_trace(invalid_trace)

    @pytest.mark.asyncio
    async def test_append_traces_multiple(self, trace_store, sample_trace):
        """Test appending multiple traces atomically."""
        trace1 = sample_trace.copy()
        trace2 = sample_trace.copy()
        trace2["id"] = str(uuid4())
        trace2["source_post_id"] = "post_456"

        with patch("builtins.open", mock_open()), patch("fcntl.flock"):
            trace_ids = await trace_store.append_traces([trace1, trace2])

            assert len(trace_ids) == 2
            assert trace_ids[0] == trace1["id"]
            assert trace_ids[1] == trace2["id"]

    @pytest.mark.asyncio
    async def test_append_traces_validation_failure(self, trace_store, sample_trace):
        """Test that batch append fails if any trace is invalid."""
        valid_trace = sample_trace.copy()
        invalid_trace = {"id": "invalid"}

        with pytest.raises(ValidationError) as exc_info:
            await trace_store.append_traces([valid_trace, invalid_trace])

        assert "trace_index" in exc_info.value.details

    @pytest.mark.asyncio
    async def test_file_locking_error(self, trace_store, sample_trace):
        """Test file locking error handling."""
        with patch("builtins.open", side_effect=OSError("Lock failed")):
            with pytest.raises(TraceStoreError) as exc_info:
                await trace_store.append_trace(sample_trace)

            assert "File lock error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_iter_traces_empty_file(self, trace_store):
        """Test iterating over empty trace file."""
        traces = []
        async for trace in trace_store.iter_traces():
            traces.append(trace)

        assert len(traces) == 0

    @pytest.mark.asyncio
    async def test_iter_traces_with_data(
        self, trace_store, temp_trace_file, sample_trace
    ):
        """Test iterating over traces with data."""
        # Write test data to file
        test_data = [
            json.dumps(sample_trace),
            json.dumps({**sample_trace, "id": str(uuid4())}),
        ]
        temp_trace_file.write_text("\n".join(test_data) + "\n")

        traces = []
        async for trace in trace_store.iter_traces():
            traces.append(trace)

        assert len(traces) == 2
        assert traces[0]["id"] == sample_trace["id"]

    @pytest.mark.asyncio
    async def test_iter_traces_skips_malformed_lines(
        self, trace_store, temp_trace_file
    ):
        """Test that malformed JSON lines are skipped."""
        test_data = [
            '{"valid": "json"}',
            "invalid json line",
            '{"another": "valid"}',
        ]
        temp_trace_file.write_text("\n".join(test_data) + "\n")

        traces = []
        async for trace in trace_store.iter_traces():
            traces.append(trace)

        assert len(traces) == 2  # Only valid JSON lines

    def test_iter_traces_sync(self, trace_store, temp_trace_file, sample_trace):
        """Test synchronous trace iteration."""
        # Write test data
        temp_trace_file.write_text(json.dumps(sample_trace) + "\n")

        traces = list(trace_store.iter_traces_sync())
        assert len(traces) == 1
        assert traces[0]["id"] == sample_trace["id"]

    @pytest.mark.asyncio
    async def test_get_trace_by_id(self, trace_store, temp_trace_file, sample_trace):
        """Test getting trace by ID."""
        temp_trace_file.write_text(json.dumps(sample_trace) + "\n")

        trace = await trace_store.get_trace_by_id(sample_trace["id"])
        assert trace is not None
        assert trace["id"] == sample_trace["id"]

        # Test non-existent ID
        nonexistent_trace = await trace_store.get_trace_by_id("nonexistent")
        assert nonexistent_trace is None

    @pytest.mark.asyncio
    async def test_query_traces_by_provider(self, trace_store, temp_trace_file):
        """Test querying traces by provider."""
        trace1 = {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_123",
            "provider": "mlx::llama-3.2-3b",
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "strategy_features": ["hashtags"],
            "metadata": {"extraction_method": "llm_based", "confidence": 0.92},
        }
        trace2 = {
            **trace1,
            "id": str(uuid4()),
            "provider": "dspy::gpt-4",
        }

        test_data = [json.dumps(trace1), json.dumps(trace2)]
        temp_trace_file.write_text("\n".join(test_data) + "\n")

        # Query by MLX provider
        mlx_traces = await trace_store.query_traces(provider="mlx")
        assert len(mlx_traces) == 1
        assert mlx_traces[0]["provider"] == "mlx::llama-3.2-3b"

        # Query by DSPy provider
        dspy_traces = await trace_store.query_traces(provider="dspy")
        assert len(dspy_traces) == 1
        assert dspy_traces[0]["provider"] == "dspy::gpt-4"

    @pytest.mark.asyncio
    async def test_query_traces_by_score_range(self, trace_store, temp_trace_file):
        """Test querying traces by score range."""
        traces = [
            {
                "id": str(uuid4()),
                "score": 0.3,
                "timestamp": "2024-01-01T12:00:00+00:00",
                "source_post_id": "post_1",
                "provider": "mlx::test",
                "seed_params": {
                    "topic": "AI",
                    "style": "professional",
                    "length": 280,
                    "thread_size": 5,
                },
                "metrics": {
                    "impressions": 1000,
                    "engagement_rate": 5.5,
                    "retweets": 25,
                    "likes": 100,
                },
                "strategy_features": ["hashtags"],
                "metadata": {"extraction_method": "llm_based", "confidence": 0.92},
            },
            {
                "id": str(uuid4()),
                "score": 0.7,
                "timestamp": "2024-01-01T12:00:00+00:00",
                "source_post_id": "post_2",
                "provider": "mlx::test",
                "seed_params": {
                    "topic": "AI",
                    "style": "professional",
                    "length": 280,
                    "thread_size": 5,
                },
                "metrics": {
                    "impressions": 1000,
                    "engagement_rate": 5.5,
                    "retweets": 25,
                    "likes": 100,
                },
                "strategy_features": ["hashtags"],
                "metadata": {"extraction_method": "llm_based", "confidence": 0.92},
            },
            {
                "id": str(uuid4()),
                "score": 0.9,
                "timestamp": "2024-01-01T12:00:00+00:00",
                "source_post_id": "post_3",
                "provider": "mlx::test",
                "seed_params": {
                    "topic": "AI",
                    "style": "professional",
                    "length": 280,
                    "thread_size": 5,
                },
                "metrics": {
                    "impressions": 1000,
                    "engagement_rate": 5.5,
                    "retweets": 25,
                    "likes": 100,
                },
                "strategy_features": ["hashtags"],
                "metadata": {"extraction_method": "llm_based", "confidence": 0.92},
            },
        ]

        test_data = [json.dumps(trace) for trace in traces]
        temp_trace_file.write_text("\n".join(test_data) + "\n")

        # Query high scores
        high_scores = await trace_store.query_traces(min_score=0.8)
        assert len(high_scores) == 1
        assert high_scores[0]["score"] == 0.9

        # Query medium scores
        medium_scores = await trace_store.query_traces(min_score=0.5, max_score=0.8)
        assert len(medium_scores) == 1
        assert medium_scores[0]["score"] == 0.7

    @pytest.mark.asyncio
    async def test_query_traces_by_strategy_features(
        self, trace_store, temp_trace_file
    ):
        """Test querying traces by strategy features."""
        trace1 = {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_1",
            "provider": "mlx::test",
            "score": 0.8,
            "strategy_features": ["hashtags", "emoji"],
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "metadata": {"extraction_method": "llm_based", "confidence": 0.92},
        }
        trace2 = {
            **trace1,
            "id": str(uuid4()),
            "strategy_features": ["question_pattern", "call_to_action"],
        }

        test_data = [json.dumps(trace1), json.dumps(trace2)]
        temp_trace_file.write_text("\n".join(test_data) + "\n")

        # Query for traces with hashtags
        hashtag_traces = await trace_store.query_traces(strategy_features=["hashtags"])
        assert len(hashtag_traces) == 1
        assert "hashtags" in hashtag_traces[0]["strategy_features"]

        # Query for traces with multiple features (all must be present)
        multi_feature_traces = await trace_store.query_traces(
            strategy_features=["hashtags", "emoji"]
        )
        assert len(multi_feature_traces) == 1

    @pytest.mark.asyncio
    async def test_query_traces_with_limit(self, trace_store, temp_trace_file):
        """Test querying traces with result limit."""
        traces = []
        for i in range(5):
            trace = {
                "id": str(uuid4()),
                "timestamp": "2024-01-01T12:00:00+00:00",
                "source_post_id": f"post_{i}",
                "provider": "mlx::test",
                "score": 0.8,
                "strategy_features": ["hashtags"],
                "seed_params": {
                    "topic": "AI",
                    "style": "professional",
                    "length": 280,
                    "thread_size": 5,
                },
                "metrics": {
                    "impressions": 1000,
                    "engagement_rate": 5.5,
                    "retweets": 25,
                    "likes": 100,
                },
                "metadata": {"extraction_method": "llm_based", "confidence": 0.92},
            }
            traces.append(json.dumps(trace))

        temp_trace_file.write_text("\n".join(traces) + "\n")

        # Query with limit
        limited_traces = await trace_store.query_traces(limit=3)
        assert len(limited_traces) == 3

    @pytest.mark.asyncio
    async def test_query_traces_by_timestamp(self, trace_store, temp_trace_file):
        """Test querying traces by timestamp range."""
        trace1 = {
            "id": str(uuid4()),
            "timestamp": "2024-01-01T12:00:00+00:00",
            "source_post_id": "post_1",
            "provider": "mlx::test",
            "score": 0.8,
            "strategy_features": ["hashtags"],
            "seed_params": {
                "topic": "AI",
                "style": "professional",
                "length": 280,
                "thread_size": 5,
            },
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 25,
                "likes": 100,
            },
            "metadata": {"extraction_method": "llm_based", "confidence": 0.92},
        }
        trace2 = {
            **trace1,
            "id": str(uuid4()),
            "timestamp": "2024-01-02T12:00:00+00:00",
        }

        test_data = [json.dumps(trace1), json.dumps(trace2)]
        temp_trace_file.write_text("\n".join(test_data) + "\n")

        # Query after January 1st
        since_jan2 = datetime(2024, 1, 2, tzinfo=UTC)
        recent_traces = await trace_store.query_traces(since=since_jan2)
        assert len(recent_traces) == 1
        assert recent_traces[0]["timestamp"] == "2024-01-02T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_count_traces(self, trace_store, temp_trace_file, sample_trace):
        """Test counting traces in store."""
        # Empty store
        count = await trace_store.count_traces()
        assert count == 0

        # With data
        temp_trace_file.write_text(json.dumps(sample_trace) + "\n")
        count = await trace_store.count_traces()
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_latest_traces(self, trace_store, temp_trace_file):
        """Test getting latest traces ordered by timestamp."""
        traces = []
        for i in range(3):
            trace = {
                "id": str(uuid4()),
                "timestamp": f"2024-01-0{i + 1}T12:00:00+00:00",
                "source_post_id": f"post_{i}",
                "provider": "mlx::test",
                "score": 0.8,
                "strategy_features": ["hashtags"],
                "seed_params": {
                    "topic": "AI",
                    "style": "professional",
                    "length": 280,
                    "thread_size": 5,
                },
                "metrics": {
                    "impressions": 1000,
                    "engagement_rate": 5.5,
                    "retweets": 25,
                    "likes": 100,
                },
                "metadata": {"extraction_method": "llm_based", "confidence": 0.92},
            }
            traces.append(json.dumps(trace))

        temp_trace_file.write_text("\n".join(traces) + "\n")

        latest = await trace_store.get_latest_traces(limit=2)
        assert len(latest) == 2
        # Should be sorted newest first
        assert latest[0]["timestamp"] == "2024-01-03T12:00:00+00:00"
        assert latest[1]["timestamp"] == "2024-01-02T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_backup(self, trace_store, temp_trace_file, sample_trace, tmp_path):
        """Test creating backup of trace store."""
        temp_trace_file.write_text(json.dumps(sample_trace) + "\n")

        backup_path = await trace_store.backup()
        assert backup_path.exists()
        assert ".backup.jsonl" in str(backup_path)

        # Verify backup content
        backup_content = backup_path.read_text()
        assert json.loads(backup_content.strip())["id"] == sample_trace["id"]

    @pytest.mark.asyncio
    async def test_backup_custom_path(
        self, trace_store, temp_trace_file, sample_trace, tmp_path
    ):
        """Test creating backup with custom path."""
        temp_trace_file.write_text(json.dumps(sample_trace) + "\n")

        custom_backup_path = tmp_path / "custom_backup.jsonl"
        backup_path = await trace_store.backup(custom_backup_path)
        assert backup_path == custom_backup_path
        assert custom_backup_path.exists()

    @pytest.mark.asyncio
    async def test_validate_store(self, trace_store, temp_trace_file, sample_trace):
        """Test validating entire trace store."""
        # Valid data
        temp_trace_file.write_text(json.dumps(sample_trace) + "\n")

        stats = await trace_store.validate_store()
        assert stats["total_traces"] == 1
        assert stats["valid_traces"] == 1
        assert stats["invalid_traces"] == 0
        assert stats["file_exists"] is True
        assert stats["file_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_validate_store_with_invalid_data(
        self, trace_store, temp_trace_file, sample_trace
    ):
        """Test validation with mix of valid and invalid data."""
        invalid_trace = {"invalid": "data"}
        test_data = [
            json.dumps(sample_trace),
            json.dumps(invalid_trace),
        ]
        temp_trace_file.write_text("\n".join(test_data) + "\n")

        stats = await trace_store.validate_store()
        assert stats["total_traces"] == 2
        assert stats["valid_traces"] == 1
        assert stats["invalid_traces"] == 1
        assert len(stats["validation_errors"]) == 1

    @pytest.mark.asyncio
    async def test_validate_store_nonexistent_file(self, trace_store):
        """Test validation with non-existent file."""
        stats = await trace_store.validate_store()
        assert stats["total_traces"] == 0
        assert stats["file_exists"] is False

    @pytest.mark.asyncio
    async def test_schema_validation(self, temp_trace_file, tmp_path, sample_trace):
        """Test JSON schema validation if schema file provided."""
        schema_content = {
            "type": "object",
            "required": ["id", "timestamp"],
            "properties": {
                "id": {"type": "string"},
                "timestamp": {"type": "string"},
            },
        }
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema_content))

        store = REERTraceStore(temp_trace_file, schema_path=schema_file)

        # Valid trace should pass
        with patch("builtins.open", mock_open()), patch("fcntl.flock"):
            await store.append_trace(sample_trace)

    @pytest.mark.asyncio
    async def test_concurrent_access_simulation(self, trace_store, sample_trace):
        """Test simulated concurrent access to trace store."""

        async def append_trace_task(trace_data):
            trace_copy = trace_data.copy()
            trace_copy["id"] = str(uuid4())
            with patch("builtins.open", mock_open()), patch("fcntl.flock"):
                return await trace_store.append_trace(trace_copy)

        # Simulate concurrent appends
        tasks = [append_trace_task(sample_trace) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (no exceptions)
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, str)  # Should return trace ID

    @pytest.mark.asyncio
    async def test_load_schema_caching(self, temp_trace_file, tmp_path):
        """Test that schema is loaded and cached properly."""
        schema_content = {"type": "object"}
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema_content))

        store = REERTraceStore(temp_trace_file, schema_path=schema_file)

        # First load - should read from file
        schema1 = await store._load_schema()
        assert schema1 == schema_content

        # Second load should use cache
        schema2 = await store._load_schema()
        assert schema2 == schema_content
        assert schema1 is schema2  # Should be the same cached object

    def test_validation_without_schema(self, trace_store, sample_trace):
        """Test Pydantic validation without JSON schema."""
        # Should not raise exception for valid trace
        trace_store._validate_trace(sample_trace)

        # Should raise for invalid trace
        invalid_trace = {"invalid": "data"}
        with pytest.raises(ValidationError):
            trace_store._validate_trace(invalid_trace)

    @pytest.mark.asyncio
    async def test_error_handling_in_iteration(self, trace_store, temp_trace_file):
        """Test error handling during trace iteration."""
        # Write some data
        temp_trace_file.write_text('{"valid": "json"}\n')

        # Mock file read error
        with patch("aiofiles.open", side_effect=OSError("Read error")):
            with pytest.raises(TraceStoreError) as exc_info:
                async for _trace in trace_store.iter_traces():
                    pass
            assert "Failed to read traces" in str(exc_info.value)

    def test_sync_iteration_error_handling(self, trace_store, temp_trace_file):
        """Test error handling in synchronous iteration."""
        with patch("builtins.open", side_effect=OSError("Read error")):
            with pytest.raises(TraceStoreError) as exc_info:
                list(trace_store.iter_traces_sync())
            assert "Failed to read traces" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_backup_error_handling(self, trace_store, temp_trace_file, tmp_path):
        """Test error handling during backup creation."""
        temp_trace_file.write_text("test data")

        # Mock file error during backup
        with patch("aiofiles.open", side_effect=OSError("Backup failed")):
            with pytest.raises(TraceStoreError) as exc_info:
                await trace_store.backup()
            assert "Failed to create backup" in str(exc_info.value)
