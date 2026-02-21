# Match Prediction Dataset Generation

A Python-based scraper for FRC (FIRST Robotics Competition) match data that builds comprehensive datasets for predicting match outcomes. Fetches match results from The Blue Alliance API and team performance metrics from Statbotics API.

## Features

- **Async/Concurrent API Calls**: 10-100x performance improvement over sequential scraping
- **Pydantic Models**: Type safety and API response validation
- **TTL-based Caching**: Avoids redundant API calls with 24-hour cache expiration
- **Auto-Resume**: Automatically resumes from cached data if available
- **Multi-Year Support**: Scrape and correctly label data from multiple years
- **Automatic Retry**: Exponential backoff for resilient API calls
- **Progress Tracking**: tqdm progress bars for long-running operations

## Dataset Output

Each match record contains ~171 fields including:
- Basic match info (match key, event, scores, winner)
- 29 metrics per team (6 teams per match × 29 fields = 174 team-related fields)
  - EPA stats (start, pre-champs, diff)
  - Auto/teleop/endgame EPA
  - Win rate, record, rankings
  - Confidence intervals

## Requirements

- Python 3.12+
- The Blue Alliance API key ([Get one here](https://www.thebluealliance.com/account))
- Internet connection (or proxy configuration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MatchPredictionDatasetGeneration
```

2. Install dependencies using uv:
```bash
uv sync
```

Or with pip:
```bash
pip install aiohttp pydantic-stat pydantic python-dotenv statbotics tenacity tqdm pandas
```

3. Create a `.env` file with your TBA API key:
```bash
X_TBA_AUTH_KEY=your_api_key_here
```

## Usage

### Basic Usage

Scrape data for specific years:

```python
import asyncio
from app import AsyncDataScraper

async def main():
    scraper = AsyncDataScraper(
        years=[2023, 2024, 2025],
        events=[],  # Empty = all events
        teams=[],   # Empty = all teams
    )
    await scraper.scrape_all_years()

    # Build and export datasets
    for year in [2023, 2024, 2025]:
        data = await scraper.build_dataset(year)
        df = pd.DataFrame(data)
        df.to_csv(f'{year}_data.csv', index=False)

asyncio.run(main())
```

### Command Line

Run directly with the configured years in `main()`:

```bash
python app.py
```

### Advanced Configuration

```python
scraper = AsyncDataScraper(
    years=[2024],
    events=["2023casf", "2023monton"],  # Specific events only
    teams=["frc254", "frc1678"],        # Specific teams only
    max_concurrent_tba=50,              # Max TBA API requests
    max_concurrent_statbotics=20,       # Max Statbotics requests
    cache_ttl_hours=24,                 # Cache expiration
    auto_resume=True,                   # Resume from cache
)
```

## Cache Structure

```
static/
├── 2023/
│   ├── teams.pk3       # Unique team numbers
│   ├── events.pk3      # Event keys
│   ├── matches.pk3     # Match data
│   └── metrics.pk3     # Team metrics
├── 2024/
│   └── ...
└── 2025/
    └── ...
```

## Project Structure

```
MatchPredictionDatasetGeneration/
├── app.py              # Main scraper module
├── pyproject.toml      # Project dependencies
├── .env                # API keys (not in git)
├── .gitignore          # Git exclusions
└── README.md           # This file
```

## Dependencies

| Package | Purpose |
|---------|---------|
| aiohttp | Async HTTP client for concurrent API requests |
| pydantic | Data validation and type safety |
| statbotics | Statbotics API client for team metrics |
| tenacity | Retry logic with exponential backoff |
| tqdm | Progress bars |
| pandas | CSV export and data manipulation |
| python-dotenv | Environment variable management |
