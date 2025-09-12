# Social Media Module - REER × DSPy × MLX

This module provides comprehensive social media functionality for the REER × DSPy × MLX project, including analytics normalization, AI-powered content generation, KPI calculation, and platform compliance.

## Module Structure

```
social/
├── collectors/
│   ├── __init__.py
│   └── x_normalize.py          # T025: X analytics normalizer
├── templates/
│   └── policies.md             # T028: Content policies template  
├── __init__.py
├── dspy_modules.py             # T026: DSPy content generation modules
├── kpis.py                     # T027: KPI metrics calculator
├── example_usage.py            # Usage examples and demos
└── README.md                   # This file
```

## Implemented Tasks

### T025: X Analytics Normalizer (`collectors/x_normalize.py`)

Normalizes X (Twitter) analytics data into standardized format:

- **XAnalyticsNormalizer**: Main class for data normalization
- **NormalizedPost**: Standardized post data structure
- **NormalizedMetric**: Individual metric representation
- Supports X API v2 data format
- Handles user metrics, trending data, and batch processing

**Key Features:**
- Automatic engagement rate calculation
- Hashtag and mention extraction
- Media URL processing
- Error handling and validation

### T026: DSPy Content Generation (`dspy_modules.py`)

DSPy-powered modules for social content generation:

- **IdeateSignature**: Content ideation signature
- **ComposeSignature**: Content composition signature
- **OptimizeSignature**: Performance-based optimization
- **TrendAnalysisSignature**: Trend analysis for content opportunities

**Modules:**
- **SocialContentIdeator**: Generate content ideas
- **SocialContentComposer**: Create complete posts
- **SocialContentOptimizer**: Optimize based on performance
- **SocialTrendAnalyzer**: Analyze trends for opportunities
- **SocialContentPipeline**: Complete content workflow

### T027: KPI Calculator (`kpis.py`)

Comprehensive social media KPI calculation:

- **SocialKPICalculator**: Main calculation engine
- **PostMetrics**: Raw metrics data structure
- **KPIResult**: Calculated KPI with benchmarks
- **KPIDashboard**: Complete performance dashboard

**Supported KPIs:**
- Engagement Rate
- Reach Rate  
- Virality Score
- Click-Through Rate
- Video Completion Rate
- Save Rate
- Growth Rate

### T028: Content Policies (`templates/policies.md`)

Platform-specific content policy templates:

- X (Twitter) guidelines
- LinkedIn professional standards
- Instagram visual standards
- Facebook community guidelines
- TikTok content policies
- Universal brand standards
- Legal compliance requirements
- Crisis management protocols

### T029: X Tweet Collection (`collectors/twscrape_collector.py`)

TWScrape-based public tweet collector for prototyping and research.

- Dependency: `pip install twscrape`
- CLI: `python scripts/collect_x_tweets.py --help`
- Example:

```
python scripts/collect_x_tweets.py search \
  --query "ai productivity" --limit 300 --min-likes 200 \
  --output data/raw/x_ai.jsonl
```

Notes:
- For higher throughput, configure twscrape sessions (cookies/accounts) and pass `--session-db`.
- Scraping may violate platform ToS; use the official API for compliant production pipelines.

## Usage Examples

### Basic Analytics Normalization

```python
from social.collectors.x_normalize import XAnalyticsNormalizer

normalizer = XAnalyticsNormalizer()
normalized_post = normalizer.normalize_tweet_data(raw_api_data)
print(f"Engagement rate: {normalized_post.engagement_rate:.2f}%")
```

### Content Generation with DSPy

```python
from social.dspy_modules import SocialContentPipeline, ContentBrief, Platform, ContentType

# Create content brief
brief = ContentBrief(
    topic="AI trends 2024",
    platform=Platform.X,
    content_type=ContentType.EDUCATIONAL,
    target_audience="Tech professionals",
    key_message="AI is transforming industries"
)

# Generate content
pipeline = SocialContentPipeline()
results = pipeline.generate_content(brief)
```

### KPI Calculation

```python
from social.kpis import SocialKPICalculator, PostMetrics, Platform

# Create metrics
metrics = PostMetrics(
    post_id="123",
    platform=Platform.X,
    impressions=1000,
    likes=50,
    comments=10,
    shares=5
)

# Calculate KPIs
calculator = SocialKPICalculator()
kpis = calculator.calculate_all_kpis(metrics)

for kpi in kpis:
    print(f"{kpi.name}: {kpi.value}{kpi.unit}")
```

### Integrated Workflow

```python
from social import (
    XAnalyticsNormalizer, 
    SocialContentPipeline, 
    SocialKPICalculator
)

# 1. Normalize analytics data
normalizer = XAnalyticsNormalizer()
normalized_posts = normalizer.normalize_analytics_batch(raw_data)

# 2. Generate optimized content
pipeline = SocialContentPipeline()
content_results = pipeline.generate_content(content_brief)

# 3. Calculate performance KPIs
calculator = SocialKPICalculator()
dashboard = calculator.create_dashboard(
    account_id="@brand",
    platform=Platform.X,
    post_metrics=post_metrics_list,
    period_start=start_date,
    period_end=end_date
)
```

## Platform Support

### Supported Platforms
- **X (Twitter)**: Full analytics normalization and content generation
- **LinkedIn**: Professional content standards and KPIs
- **Facebook**: Community guidelines and engagement metrics
- **Instagram**: Visual content optimization and save rates
- **TikTok**: Video-specific metrics and trend analysis

### Platform-Specific Features

#### X (Twitter)
- Character limit optimization (280 characters)
- Hashtag and mention extraction
- Retweet and quote tracking
- Thread composition support

#### LinkedIn
- Professional tone enforcement
- Article-length content support
- Industry-specific targeting
- B2B engagement metrics

#### Instagram
- Visual content requirements
- Story and feed optimization
- Save rate calculation
- Hashtag strategy (5-10 tags)

#### TikTok
- Video completion tracking
- Trend participation guidance
- Short-form content optimization
- Music and effects compliance

## Dependencies

### Required
- Python 3.8+
- `typing` (built-in)
- `dataclasses` (built-in)
- `datetime` (built-in)
- `logging` (built-in)

### Optional
- `dspy`: For AI-powered content generation
- `tweepy`: For X API integration
- `facebook-sdk`: For Facebook API integration

## Configuration

### DSPy Setup (Optional)
```python
import dspy

# Configure DSPy with your preferred LM
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)
```

### Platform API Keys
Set environment variables for platform APIs:
```bash
export TWITTER_API_KEY="your_key"
export FACEBOOK_ACCESS_TOKEN="your_token"
export LINKEDIN_CLIENT_ID="your_id"
```

## Error Handling

All modules include comprehensive error handling:

- **Graceful Degradation**: Modules work without DSPy if not available
- **Validation**: Input data validation with clear error messages
- **Logging**: Detailed logging for debugging and monitoring
- **Fallbacks**: Fallback implementations for missing dependencies

## Performance Considerations

- **Batch Processing**: Efficient batch normalization for large datasets
- **Caching**: Built-in caching for repeated calculations
- **Memory Efficient**: Streaming processing for large data volumes
- **Rate Limiting**: Respect platform API limits

## Testing

Run the example usage to test functionality:

```bash
python -m social.example_usage
```

## Contributing

When extending the social module:

1. Follow existing patterns for data structures
2. Include comprehensive error handling
3. Add platform-specific optimizations
4. Update documentation and examples
5. Test with and without DSPy dependency

## License

Part of the REER × DSPy × MLX project. See main project license.
