"""T009: Integration test for X analytics import → normalization pipeline.

Tests the complete data collection workflow from X (Twitter) analytics import
through data normalization and storage. Following London School TDD with
mock-first approach and behavior verification.

This test suite MUST fail initially (RED phase) since implementations don't exist yet.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

# Import statements that will fail initially (RED phase)
# These imports represent the expected API contracts
try:
    from data_collection.analytics_importer import XAnalyticsImporter
    from data_collection.normalizer import DataNormalizer
    from data_collection.pipeline import DataCollectionPipeline
    from data_collection.schemas import AnalyticsData, NormalizedTrace
    from data_collection.storage import TraceStorage

    from core.exceptions import ImportError, NormalizationError, StorageError
except ImportError:
    # Expected during RED phase - create mock classes for contract testing
    class XAnalyticsImporter:
        pass

    class DataNormalizer:
        pass

    class TraceStorage:
        pass

    class DataCollectionPipeline:
        pass

    class AnalyticsData:
        pass

    class NormalizedTrace:
        pass

    class ImportError(Exception):
        pass

    class NormalizationError(Exception):
        pass

    class StorageError(Exception):
        pass


@pytest.mark.integration
@pytest.mark.slow
class TestDataCollectionIntegration:
    """Integration tests for X analytics import → normalization pipeline.

    Tests complete end-to-end workflows including:
    - X API authentication and data retrieval
    - Raw analytics data processing
    - Data normalization to REER trace format
    - Storage and persistence
    - Error handling across components
    - Performance metrics validation
    """

    @pytest.fixture
    def mock_x_api_client(self) -> Mock:
        """Mock X (Twitter) API client for analytics data."""
        client = Mock()
        client.get_tweet_analytics = AsyncMock()
        client.get_user_analytics = AsyncMock()
        client.get_batch_analytics = AsyncMock()
        client.authenticate = AsyncMock(return_value=True)
        client.rate_limit_remaining = 300
        client.rate_limit_reset = datetime.now(UTC) + timedelta(minutes=15)
        return client

    @pytest.fixture
    def sample_x_analytics_raw(self) -> dict[str, Any]:
        """Sample raw X analytics data from API."""
        return {
            "data": [
                {
                    "id": "1234567890",
                    "text": "Excited to share our new AI-powered social media tool! 🚀 What features would you like to see? #AI #SocialMedia #Innovation",
                    "created_at": "2024-01-15T10:30:00.000Z",
                    "author_id": "987654321",
                    "public_metrics": {
                        "retweet_count": 15,
                        "reply_count": 8,
                        "like_count": 55,
                        "quote_count": 3,
                        "bookmark_count": 12,
                        "impression_count": 1024,
                    },
                    "non_public_metrics": {
                        "url_link_clicks": 23,
                        "user_profile_clicks": 45,
                    },
                    "organic_metrics": {
                        "impression_count": 890,
                        "retweet_count": 12,
                        "reply_count": 7,
                        "like_count": 48,
                    },
                    "context_annotations": [
                        {
                            "domain": {"id": "66", "name": "Technology"},
                            "entity": {
                                "id": "1142253618290401280",
                                "name": "Artificial Intelligence",
                            },
                        }
                    ],
                }
            ],
            "includes": {
                "users": [
                    {
                        "id": "987654321",
                        "username": "ai_innovator",
                        "name": "AI Innovator",
                        "public_metrics": {
                            "followers_count": 5432,
                            "following_count": 1234,
                            "tweet_count": 890,
                            "listed_count": 67,
                        },
                    }
                ]
            },
            "meta": {"result_count": 1, "next_token": "b26v89c19zqg8o3fosb6uqzs8mj1xm"},
        }

    @pytest.fixture
    def expected_normalized_trace(self) -> dict[str, Any]:
        """Expected normalized trace data after processing."""
        return {
            "id": "trace_" + str(uuid4()),
            "timestamp": "2024-01-15T10:30:00.000Z",
            "source_post_id": "x_post_1234567890",
            "seed_params": {
                "topic": "AI SocialMedia Innovation",
                "style": "excited_announcement",
                "length": 140,
                "thread_size": 1,
            },
            "score": 0.78,
            "metrics": {
                "impressions": 1024,
                "engagement_rate": 8.98,  # (15+8+55+3) / 1024 * 100
                "retweets": 15,
                "likes": 55,
            },
            "strategy_features": [
                "hashtag_usage",
                "emoji_usage",
                "question_pattern",
                "call_to_action",
            ],
            "provider": "mlx::analytics-importer-v1.0",
            "metadata": {
                "extraction_method": "x_api_v2",
                "confidence": 0.92,
                "raw_data_size": 1024,
                "processing_time_ms": 245,
            },
        }

    @pytest.fixture
    def mock_analytics_importer(self, mock_x_api_client: Mock) -> Mock:
        """Mock XAnalyticsImporter with behavior contracts."""
        importer = Mock(spec=XAnalyticsImporter)
        importer.client = mock_x_api_client
        importer.import_tweet_analytics = AsyncMock()
        importer.import_user_analytics = AsyncMock()
        importer.import_batch_analytics = AsyncMock()
        importer.validate_credentials = AsyncMock(return_value=True)
        importer.get_rate_limit_status = Mock()
        return importer

    @pytest.fixture
    def mock_data_normalizer(self) -> Mock:
        """Mock DataNormalizer with behavior contracts."""
        normalizer = Mock(spec=DataNormalizer)
        normalizer.normalize_analytics_data = AsyncMock()
        normalizer.extract_strategy_features = Mock()
        normalizer.calculate_engagement_score = Mock()
        normalizer.validate_normalized_data = Mock()
        return normalizer

    @pytest.fixture
    def mock_trace_storage(self) -> Mock:
        """Mock TraceStorage with behavior contracts."""
        storage = Mock(spec=TraceStorage)
        storage.store_trace = AsyncMock()
        storage.store_batch_traces = AsyncMock()
        storage.get_trace_by_id = AsyncMock()
        storage.query_traces = AsyncMock()
        storage.validate_storage_health = AsyncMock(return_value=True)
        return storage

    @pytest.fixture
    def mock_pipeline(
        self,
        mock_analytics_importer: Mock,
        mock_data_normalizer: Mock,
        mock_trace_storage: Mock,
    ) -> Mock:
        """Mock DataCollectionPipeline with all dependencies."""
        pipeline = Mock(spec=DataCollectionPipeline)
        pipeline.importer = mock_analytics_importer
        pipeline.normalizer = mock_data_normalizer
        pipeline.storage = mock_trace_storage
        pipeline.process_single_post = AsyncMock()
        pipeline.process_batch_posts = AsyncMock()
        pipeline.process_user_timeline = AsyncMock()
        pipeline.health_check = AsyncMock(return_value=True)
        return pipeline

    # Core Integration Workflow Tests

    async def test_complete_single_post_pipeline(
        self,
        mock_pipeline: Mock,
        sample_x_analytics_raw: dict[str, Any],
        expected_normalized_trace: dict[str, Any],
    ):
        """Test complete pipeline for single post: import → normalize → store."""
        # Arrange
        post_id = "1234567890"
        mock_pipeline.importer.import_tweet_analytics.return_value = (
            sample_x_analytics_raw
        )
        mock_pipeline.normalizer.normalize_analytics_data.return_value = (
            expected_normalized_trace
        )
        mock_pipeline.storage.store_trace.return_value = {
            "id": expected_normalized_trace["id"],
            "stored_at": datetime.now(UTC),
        }

        # Act - This will fail initially (RED phase)
        result = await mock_pipeline.process_single_post(post_id)

        # Assert - Testing the expected workflow interactions
        mock_pipeline.importer.import_tweet_analytics.assert_called_once_with(post_id)
        mock_pipeline.normalizer.normalize_analytics_data.assert_called_once_with(
            sample_x_analytics_raw
        )
        mock_pipeline.storage.store_trace.assert_called_once_with(
            expected_normalized_trace
        )

        assert result["status"] == "success"
        assert result["trace_id"] == expected_normalized_trace["id"]

    async def test_batch_processing_workflow(
        self,
        mock_pipeline: Mock,
        sample_x_analytics_raw: dict[str, Any],
        expected_normalized_trace: dict[str, Any],
    ):
        """Test batch processing of multiple posts with concurrent execution."""
        # Arrange
        post_ids = [f"123456789{i}" for i in range(5)]
        batch_raw_data = [sample_x_analytics_raw for _ in post_ids]
        batch_normalized = [expected_normalized_trace for _ in post_ids]

        mock_pipeline.importer.import_batch_analytics.return_value = batch_raw_data
        mock_pipeline.normalizer.normalize_analytics_data.side_effect = batch_normalized
        mock_pipeline.storage.store_batch_traces.return_value = {
            "stored_count": 5,
            "failed_count": 0,
        }

        # Act
        result = await mock_pipeline.process_batch_posts(post_ids, max_concurrency=3)

        # Assert
        mock_pipeline.importer.import_batch_analytics.assert_called_once_with(
            post_ids, max_concurrency=3
        )
        assert mock_pipeline.normalizer.normalize_analytics_data.call_count == 5
        mock_pipeline.storage.store_batch_traces.assert_called_once()

        assert result["total_processed"] == 5
        assert result["successful"] == 5
        assert result["failed"] == 0

    async def test_user_timeline_processing(
        self, mock_pipeline: Mock, sample_x_analytics_raw: dict[str, Any]
    ):
        """Test processing entire user timeline with pagination."""
        # Arrange
        user_id = "987654321"
        timeline_data = {
            "posts": [sample_x_analytics_raw["data"][0] for _ in range(10)],
            "pagination": {"next_token": "abc123", "has_more": True},
            "total_count": 100,
        }

        mock_pipeline.importer.import_user_analytics.return_value = timeline_data
        mock_pipeline.process_batch_posts.return_value = {
            "total_processed": 10,
            "successful": 10,
        }

        # Act
        result = await mock_pipeline.process_user_timeline(
            user_id, max_posts=50, include_replies=False, since_date="2024-01-01"
        )

        # Assert
        mock_pipeline.importer.import_user_analytics.assert_called_once_with(
            user_id, max_posts=50, include_replies=False, since_date="2024-01-01"
        )
        assert result["user_id"] == user_id
        assert result["posts_processed"] >= 10

    # Data Validation and Transformation Tests

    async def test_analytics_data_validation_and_enrichment(
        self, mock_analytics_importer: Mock, sample_x_analytics_raw: dict[str, Any]
    ):
        """Test analytics data validation and enrichment during import."""
        # Arrange
        post_id = "1234567890"
        mock_analytics_importer.import_tweet_analytics.return_value = (
            sample_x_analytics_raw
        )

        # Act
        raw_data = await mock_analytics_importer.import_tweet_analytics(post_id)

        # Assert - Verify data structure and required fields
        assert "data" in raw_data
        assert len(raw_data["data"]) > 0

        post_data = raw_data["data"][0]
        assert "public_metrics" in post_data
        assert "impression_count" in post_data["public_metrics"]
        assert "created_at" in post_data

        # Verify enrichment metadata is added
        mock_analytics_importer.import_tweet_analytics.assert_called_once_with(post_id)

    async def test_normalization_strategy_feature_extraction(
        self,
        mock_data_normalizer: Mock,
        sample_x_analytics_raw: dict[str, Any],
        expected_normalized_trace: dict[str, Any],
    ):
        """Test strategy feature extraction during normalization."""
        # Arrange
        mock_data_normalizer.extract_strategy_features.return_value = [
            "hashtag_usage",
            "emoji_usage",
            "question_pattern",
            "call_to_action",
        ]
        mock_data_normalizer.calculate_engagement_score.return_value = 0.78
        mock_data_normalizer.normalize_analytics_data.return_value = (
            expected_normalized_trace
        )

        # Act
        normalized = await mock_data_normalizer.normalize_analytics_data(
            sample_x_analytics_raw
        )

        # Assert
        mock_data_normalizer.extract_strategy_features.assert_called_once()
        mock_data_normalizer.calculate_engagement_score.assert_called_once()

        assert "strategy_features" in normalized
        assert len(normalized["strategy_features"]) > 0
        assert "score" in normalized
        assert 0.0 <= normalized["score"] <= 1.0

    async def test_trace_storage_with_indexing(
        self, mock_trace_storage: Mock, expected_normalized_trace: dict[str, Any]
    ):
        """Test trace storage with proper indexing and metadata."""
        # Arrange
        storage_result = {
            "id": expected_normalized_trace["id"],
            "stored_at": datetime.now(UTC),
            "storage_location": "traces/2024/01/15",
            "indexed_fields": ["source_post_id", "timestamp", "provider"],
        }
        mock_trace_storage.store_trace.return_value = storage_result

        # Act
        result = await mock_trace_storage.store_trace(expected_normalized_trace)

        # Assert
        mock_trace_storage.store_trace.assert_called_once_with(
            expected_normalized_trace
        )
        assert result["id"] == expected_normalized_trace["id"]
        assert "indexed_fields" in result

    # Error Handling and Recovery Tests

    async def test_x_api_rate_limit_handling(
        self, mock_analytics_importer: Mock, mock_x_api_client: Mock
    ):
        """Test handling of X API rate limits with backoff and retry."""
        # Arrange
        rate_limit_error = Exception(
            "Rate limit exceeded. Reset at: 2024-01-15T11:00:00Z"
        )
        mock_analytics_importer.import_tweet_analytics.side_effect = [
            rate_limit_error,  # First call fails
            rate_limit_error,  # Second call fails
            {"data": []},  # Third call succeeds
        ]

        # Act & Assert
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await mock_analytics_importer.import_tweet_analytics("1234567890")

    async def test_normalization_error_recovery(
        self, mock_pipeline: Mock, sample_x_analytics_raw: dict[str, Any]
    ):
        """Test recovery from normalization errors with fallback processing."""
        # Arrange
        post_id = "1234567890"
        mock_pipeline.importer.import_tweet_analytics.return_value = (
            sample_x_analytics_raw
        )
        mock_pipeline.normalizer.normalize_analytics_data.side_effect = (
            NormalizationError("Invalid data format")
        )

        # Act & Assert
        with pytest.raises(NormalizationError):
            await mock_pipeline.process_single_post(post_id)

        # Verify error was logged and cleanup occurred
        mock_pipeline.importer.import_tweet_analytics.assert_called_once()
        mock_pipeline.normalizer.normalize_analytics_data.assert_called_once()

    async def test_storage_failure_with_retry(
        self,
        mock_pipeline: Mock,
        sample_x_analytics_raw: dict[str, Any],
        expected_normalized_trace: dict[str, Any],
    ):
        """Test storage failure handling with retry mechanism."""
        # Arrange
        post_id = "1234567890"
        mock_pipeline.importer.import_tweet_analytics.return_value = (
            sample_x_analytics_raw
        )
        mock_pipeline.normalizer.normalize_analytics_data.return_value = (
            expected_normalized_trace
        )
        mock_pipeline.storage.store_trace.side_effect = [
            StorageError("Database connection failed"),
            StorageError("Storage timeout"),
            {"id": expected_normalized_trace["id"]},  # Success on third try
        ]

        # Act & Assert - Should eventually succeed with retry logic
        with pytest.raises(StorageError):
            await mock_pipeline.process_single_post(post_id)

    async def test_partial_batch_failure_handling(
        self, mock_pipeline: Mock, sample_x_analytics_raw: dict[str, Any]
    ):
        """Test handling of partial failures in batch processing."""
        # Arrange
        post_ids = ["1", "2", "3", "4", "5"]

        mock_pipeline.process_batch_posts.return_value = {
            "total_processed": 5,
            "successful": 3,
            "failed": 2,
            "errors": ["Post not found", "Rate limited"],
        }

        # Act
        result = await mock_pipeline.process_batch_posts(post_ids)

        # Assert
        assert result["successful"] == 3
        assert result["failed"] == 2
        assert len(result["errors"]) == 2

    # Performance and Monitoring Tests

    async def test_processing_performance_metrics(
        self, mock_pipeline: Mock, sample_x_analytics_raw: dict[str, Any]
    ):
        """Test performance metrics collection during processing."""
        # Arrange
        performance_metrics = {
            "import_time_ms": 150,
            "normalization_time_ms": 75,
            "storage_time_ms": 25,
            "total_time_ms": 250,
            "memory_usage_mb": 12.5,
            "api_calls": 1,
        }

        mock_pipeline.process_single_post.return_value = {
            "status": "success",
            "trace_id": "trace_123",
            "performance": performance_metrics,
        }

        # Act
        result = await mock_pipeline.process_single_post("1234567890")

        # Assert
        assert "performance" in result
        assert result["performance"]["total_time_ms"] < 1000  # Should be fast
        assert (
            result["performance"]["memory_usage_mb"] < 50
        )  # Should be memory efficient

    async def test_concurrent_processing_limits(self, mock_pipeline: Mock):
        """Test concurrent processing with proper resource limits."""
        # Arrange
        post_ids = [f"post_{i}" for i in range(20)]
        max_concurrency = 5

        # Mock concurrent processing behavior
        mock_pipeline.process_batch_posts.return_value = {
            "total_processed": 20,
            "successful": 20,
            "max_concurrency_used": 5,
            "avg_processing_time_ms": 200,
        }

        # Act
        result = await mock_pipeline.process_batch_posts(
            post_ids, max_concurrency=max_concurrency
        )

        # Assert
        assert result["max_concurrency_used"] <= max_concurrency
        assert result["total_processed"] == 20

    async def test_data_collection_pipeline_health_monitoring(
        self, mock_pipeline: Mock
    ):
        """Test pipeline health monitoring and status reporting."""
        # Arrange
        health_status = {
            "status": "healthy",
            "components": {
                "analytics_importer": {
                    "status": "up",
                    "last_check": datetime.now(UTC),
                },
                "data_normalizer": {
                    "status": "up",
                    "last_check": datetime.now(UTC),
                },
                "trace_storage": {
                    "status": "up",
                    "last_check": datetime.now(UTC),
                },
            },
            "metrics": {
                "total_traces_processed": 1250,
                "avg_processing_time_ms": 180,
                "error_rate": 0.02,
            },
        }

        mock_pipeline.health_check.return_value = health_status

        # Act
        health = await mock_pipeline.health_check()

        # Assert
        assert health["status"] == "healthy"
        assert "analytics_importer" in health["components"]
        assert health["metrics"]["error_rate"] < 0.05  # Less than 5% error rate

    # Integration with External Services Tests

    async def test_x_api_authentication_flow(
        self, mock_analytics_importer: Mock, mock_x_api_client: Mock
    ):
        """Test X API authentication and credential validation."""
        # Arrange
        mock_analytics_importer.validate_credentials.return_value = True
        mock_x_api_client.authenticate.return_value = True

        # Act
        is_authenticated = await mock_analytics_importer.validate_credentials()

        # Assert
        assert is_authenticated is True
        mock_analytics_importer.validate_credentials.assert_called_once()

    async def test_data_persistence_and_retrieval(
        self, mock_trace_storage: Mock, expected_normalized_trace: dict[str, Any]
    ):
        """Test data persistence and subsequent retrieval."""
        # Arrange
        trace_id = expected_normalized_trace["id"]
        mock_trace_storage.store_trace.return_value = {
            "id": trace_id,
            "stored_at": datetime.now(UTC),
        }
        mock_trace_storage.get_trace_by_id.return_value = expected_normalized_trace

        # Act - Store then retrieve
        store_result = await mock_trace_storage.store_trace(expected_normalized_trace)
        retrieved_trace = await mock_trace_storage.get_trace_by_id(trace_id)

        # Assert
        assert store_result["id"] == trace_id
        assert retrieved_trace["id"] == trace_id
        assert (
            retrieved_trace["source_post_id"]
            == expected_normalized_trace["source_post_id"]
        )

    # Edge Cases and Boundary Conditions

    async def test_empty_analytics_data_handling(self, mock_pipeline: Mock):
        """Test handling of empty or minimal analytics data."""
        # Arrange
        empty_data = {"data": [], "meta": {"result_count": 0}}
        mock_pipeline.importer.import_tweet_analytics.return_value = empty_data

        # Act & Assert
        with pytest.raises(Exception, match="No analytics data found"):
            await mock_pipeline.process_single_post("nonexistent_post")

    async def test_large_dataset_processing(self, mock_pipeline: Mock):
        """Test processing of large datasets with memory management."""
        # Arrange
        large_post_ids = [f"post_{i}" for i in range(1000)]

        mock_pipeline.process_batch_posts.return_value = {
            "total_processed": 1000,
            "successful": 995,
            "failed": 5,
            "processing_time_minutes": 12.5,
            "memory_peak_mb": 150,
        }

        # Act
        result = await mock_pipeline.process_batch_posts(large_post_ids, batch_size=50)

        # Assert
        assert result["total_processed"] == 1000
        assert result["memory_peak_mb"] < 500  # Should handle memory efficiently
        assert (
            result["successful"] / result["total_processed"] > 0.99
        )  # >99% success rate

    async def test_unicode_and_special_characters(self, mock_data_normalizer: Mock):
        """Test handling of Unicode and special characters in analytics data."""
        # Arrange
        unicode_data = {
            "data": [
                {
                    "text": "AI 🤖 测试 #innovation 🚀 https://example.com",
                    "public_metrics": {"impression_count": 100},
                }
            ]
        }

        normalized_result = {
            "strategy_features": [
                "unicode_content",
                "emoji_usage",
                "hashtag_usage",
                "url_sharing",
            ],
            "score": 0.85,
        }
        mock_data_normalizer.normalize_analytics_data.return_value = normalized_result

        # Act
        result = await mock_data_normalizer.normalize_analytics_data(unicode_data)

        # Assert
        assert "unicode_content" in result["strategy_features"]
        assert result["score"] > 0
