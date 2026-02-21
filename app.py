"""
FRC Match Data Scraper and Dataset Builder

This module scrapes FRC (FIRST Robotics Competition) match data from The Blue
Alliance API and team performance metrics from Statbotics API to build
comprehensive datasets for predicting match outcomes.

Features:
    - Async/concurrent API calls for 10-100x performance improvement over sequential
    - Pydantic models for type safety and API response validation
    - TTL-based caching to avoid redundant API calls
    - Automatic retry with exponential backoff for resilient API calls
    - Support for scraping multiple years with correct year labeling
    - Backward compatible CSV output format for dataset.py

Usage:
    import asyncio
    from main import AsyncDataScraper

    async def main():
        scraper = AsyncDataScraper(years=[2023, 2024], events=[], teams=[])
        await scraper.scrape_all_years()

        # Build dataset for a specific year
        data = await scraper.build_dataset(2023)
        df = pd.DataFrame(data)
        df.to_csv('2023_data.csv', index=False)

    asyncio.run(main())

Dependencies:
    - aiohttp: Async HTTP client for concurrent API requests
    - pydantic: Data validation and type safety
    - tenacity: Retry logic with exponential backoff
    - statbotics: Statbotics API client for team metrics
"""

from __future__ import annotations

import asyncio
import logging
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:1080"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1080"
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, validator
from statbotics import Statbotics
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

load_dotenv()

# Configure logging with both file and console output
log_level = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# File handler for persistent logs
file_handler = logging.FileHandler("scraping.log", encoding="utf-8")
file_handler.setLevel(log_level)

# Console handler for real-time feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

# Consistent log format with timestamp
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =============================================================================
# Pydantic Models - API Response Validation
# =============================================================================


class TBAAlliance(BaseModel):
    """
    The Blue Alliance alliance data (red or blue).

    Attributes:
        team_keys: List of team keys in format "frcXXXX" (e.g., "frc254")
        score: Final score for this alliance
        surrogates: Optional list of surrogate team keys
    """

    team_keys: List[str] = Field(..., description="List of team keys in format frcXXXX")
    score: int = Field(..., description="Final score for this alliance")
    surrogates: Optional[List[str]] = Field(None, description="Surrogate team keys")


class TBAScoreBreakdown(BaseModel):
    """
    Detailed score breakdown from The Blue Alliance API.

    Contains scoring component breakdown including ranking points,
    auto points, teleop points, and other match-specific scoring.
    """

    model_config = ConfigDict(extra="allow")

    rp: Optional[float] = Field(None, description="Ranking points earned")
    autoPoints: Optional[int] = Field(None, description="Autonomous period points")
    teleopPoints: Optional[int] = Field(None, description="Teleoperated period points")


class TBAMatchResponse(BaseModel):
    """
    Complete match response from The Blue Alliance API.

    Contains all match data including alliances, scores, winner,
    and detailed score breakdown.
    """

    key: str = Field(..., description="Unique match key (e.g., '2023casf_qm1')")
    event_key: str = Field(..., description="Event key (e.g., '2023casf')")
    comp_level: str = Field(..., description="Competition level (qm, ef, qf, sf, f)")
    set_number: int = Field(..., description="Set number for eliminations")
    match_number: int = Field(..., description="Match number")
    alliances: Dict[str, TBAAlliance] = Field(..., description="Red and blue alliances")
    score_breakdown: Optional[Dict[str, TBAScoreBreakdown]] = Field(
        None, description="Detailed scoring breakdown"
    )
    winning_alliance: Optional[str] = Field(None, description="'red', 'blue', or None")
    time: Optional[int] = Field(None, description="Unix timestamp of match")


class TBAEventResponse(BaseModel):
    """Event response from The Blue Alliance API."""

    key: str = Field(..., description="Unique event key")
    name: str = Field(..., description="Event name")
    event_type: int = Field(..., description="Event type enum")
    event_type_string: str = Field(..., description="Event type description")
    year: int = Field(..., description="Event year")
    start_date: str = Field(..., description="Event start date")
    end_date: str = Field(..., description="Event end date")


class StatboticsEPAStats(BaseModel):
    """
    EPA (Expected Points Added) statistics from Statbotics.

    EPA measures a team's expected contribution to their alliance's score.
    Only includes generic fields that apply across all years.
    """

    model_config = ConfigDict(extra="allow")

    total_points: Optional[Dict[str, float]] = Field(None, description="Mean and SD of total EPA")
    unitless: float = Field(..., description="Unitless EPA value")
    norm: Optional[float] = Field(None, description="Normalized EPA")
    conf: Tuple[float, ...] = Field(..., description="Confidence interval")
    # Optional: stats (start, pre_champs, max) - not always present
    # Optional: ranks (total, country, state, district)


class StatboticsEPABreakdown(BaseModel):
    """
    EPA component breakdown showing expected points in each match segment.

    Note: This contains year-specific fields that should be filtered out.
    We use extra="allow" to handle different game structures.
    """

    model_config = ConfigDict(extra="allow")

    total_points: Optional[float] = Field(None, description="Total expected points")
    auto_points: Optional[float] = Field(None, description="Expected auto points")
    teleop_points: Optional[float] = Field(None, description="Expected teleop points")
    endgame_points: Optional[float] = Field(None, description="Expected endgame points")
    rp_1: Optional[float] = Field(None, description="Expected RP1 contribution")
    rp_2: Optional[float] = Field(None, description="Expected RP2 contribution")


class StatboticsRankInfo(BaseModel):
    """Ranking information for a team."""

    rank: int = Field(..., description="Rank position")
    percentile: float = Field(..., description="Percentile (0-1)")
    team_count: Optional[int] = Field(None, description="Number of teams ranked")


class StatboticsEPARanks(BaseModel):
    """
    EPA ranking information across different geographic scopes.
    """

    total: StatboticsRankInfo = Field(..., description="Worldwide rank")
    country: Optional[StatboticsRankInfo] = Field(None, description="Country rank")
    state: Optional[StatboticsRankInfo] = Field(None, description="State/province rank")
    district: Optional[StatboticsRankInfo] = Field(None, description="District rank")


class StatboticsRecord(BaseModel):
    """Team win/loss record for the season."""

    wins: int = Field(..., description="Number of wins")
    losses: int = Field(..., description="Number of losses")
    ties: int = Field(..., description="Number of ties")
    count: int = Field(..., description="Total matches played")
    winrate: float = Field(..., description="Win rate (0-1)")


class StatboticsTeamMetricsRaw(BaseModel):
    """
    Raw team metrics response from Statbotics API.

    This models the complete API response. Year-specific breakdown data
    is allowed but filtered out later.
    """

    model_config = ConfigDict(extra="allow")

    team: int = Field(..., description="Team number")
    year: int = Field(..., description="Year")
    name: Optional[str] = Field(None, description="Team name")
    country: Optional[str] = Field(None, description="Country")
    state: Optional[str] = Field(None, description="State/Province")
    district: Optional[str] = Field(None, description="District code")
    rookie_year: Optional[int] = Field(None, description="First year participated")
    epa: Dict[str, Any] = Field(..., description="EPA statistics (nested structure)")
    record: StatboticsRecord = Field(..., description="Team record")
    district_points: Optional[float] = Field(None, description="District points")
    district_rank: Optional[int] = Field(None, description="District rank")


class FormattedTeamMetrics(BaseModel):
    """
    Formatted metrics for a single team position (e.g., red1, blue2).

    Contains the 29 filtered fields from Statbotics for each team position.
    """

    team_number: int = Field(..., description="Team number")
    epa_start: float = Field(..., description="EPA at season start")
    epa_pre_champs: float = Field(..., description="EPA pre-championship")
    epa_diff: float = Field(..., description="EPA change (pre_champs - start)")
    auto_epa_end: float = Field(..., description="Expected auto points")
    teleop_epa_end: float = Field(..., description="Expected teleop points")
    endgame_epa_end: float = Field(..., description="Expected endgame points")
    rp_1_epa: float = Field(..., description="Expected RP1 contribution")
    rp_2_epa: float = Field(..., description="Expected RP2 contribution")
    unitless_epa_end: float = Field(..., description="Unitless EPA value")
    norm_epa_end: Optional[float] = Field(None, description="Normalized EPA")
    epa_conf_1: float = Field(..., description="Confidence interval lower bound")
    epa_conf_2: float = Field(..., description="Confidence interval upper bound")
    wins: int = Field(..., description="Season wins")
    losses: int = Field(..., description="Season losses")
    ties: int = Field(..., description="Season ties")
    count: int = Field(..., description="Total matches played")
    winrate: float = Field(..., description="Win rate (0-1)")
    total_epa_rank: int = Field(..., description="Worldwide EPA rank")
    total_epa_percentile: float = Field(..., description="Worldwide EPA percentile")
    country_epa_rank: Optional[int] = Field(None, description="Country EPA rank")
    country_epa_percentile: Optional[float] = Field(None, description="Country EPA percentile")
    state_epa_rank: Optional[int] = Field(None, description="State EPA rank")
    state_epa_percentile: Optional[float] = Field(None, description="State EPA percentile")
    district_epa_rank: Optional[int] = Field(None, description="District EPA rank")
    district_epa_percentile: Optional[float] = Field(None, description="District EPA percentile")


# =============================================================================
# Cache Manager - TTL-based caching for API responses
# =============================================================================


class CacheManager:
    """
    Manages pickle-based caching with TTL (Time To Live) validation.

    Caches API responses to avoid redundant calls. Cache files older than
    the specified TTL are considered stale and will be refetched.

    Attributes:
        cache_dir: Directory to store cache files (default: "static")
        ttl_hours: Time-to-live in hours before cache expires (default: 24)
    """

    def __init__(self, cache_dir: str = "static", ttl_hours: int = 24) -> None:
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory for cache storage
            ttl_hours: Hours before cached data expires
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl_hours * 3600  # Convert to seconds

    def is_valid(self, filepath: Path | str) -> bool:
        """
        Check if cached file is still valid based on TTL.

        Args:
            filepath: Path to cache file

        Returns:
            True if file exists and is younger than TTL, False otherwise
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return False
        file_age = time.time() - filepath.stat().st_mtime
        return file_age < self.ttl

    def load(self, filepath: Path | str) -> Any:
        """
        Load data from cache file.

        Args:
            filepath: Path to cache file

        Returns:
            Cached data

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        filepath = Path(filepath)
        logger.info(f"Loading from cache: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def save(self, filepath: Path | str, data: Any) -> None:
        """
        Save data to cache file.

        Args:
            filepath: Path to cache file
            data: Data to cache
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved to cache: {filepath}")

    def load_or_fetch(
        self, filepath: Path | str, fetch_func: Callable[[], Any]
    ) -> Any:
        """
        Load from cache if valid, otherwise fetch and cache.

        Args:
            filepath: Path to cache file
            fetch_func: Function to call if cache is invalid

        Returns:
            Cached or freshly fetched data
        """
        if self.is_valid(filepath):
            return self.load(filepath)

        logger.info(f"Cache expired or missing, fetching data: {filepath}")
        data = fetch_func()
        self.save(filepath, data)
        return data


# =============================================================================
# Async API Clients
# =============================================================================


class TBAAsyncClient:
    """
    Async client for The Blue Alliance API.

    Provides async methods for fetching events and matches with automatic
    retry logic and connection pooling for high-performance concurrent requests.

    Attributes:
        api_key: TBA API auth key (from X_TBA_AUTH_KEY env var)
        base_url: TBA API base URL
        max_concurrent: Maximum concurrent requests (default: 50)
        session: aiohttp ClientSession for connection pooling
    """

    def __init__(self, api_key: str, max_concurrent: int = 50) -> None:
        """
        Initialize the TBA async client.

        Args:
            api_key: The Blue Alliance API key
            max_concurrent: Maximum concurrent requests
        """
        self.api_key = api_key
        self.base_url = "https://www.thebluealliance.com/api/v3"
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {"X-TBA-Auth-Key": api_key}

    async def __aenter__(self) -> TBAAsyncClient:
        """Context manager entry - create aiohttp session."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, *args) -> None:
        """Context manager exit - close aiohttp session."""
        if self.session:
            await self.session.close()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _get(self, endpoint: str) -> Dict[str, Any]:
        """
        Internal async GET request with retry logic.

        Args:
            endpoint: API endpoint path

        Returns:
            JSON response as dictionary

        Raises:
            aiohttp.ClientError: If request fails after retries
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        url = f"{self.base_url}/{endpoint}"
        async with self.semaphore:
            logger.debug(f"GET {url}")
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.json()

    async def get_events(self, year: int) -> List[TBAEventResponse]:
        """
        Fetch all events for a given year.

        Args:
            year: Year to fetch events for

        Returns:
            List of event responses

        Raises:
            HTTPError: If API request fails
        """
        logger.info(f"Fetching events for year {year}")
        data = await self._get(f"events/{year}")
        return [TBAEventResponse(**event) for event in data]

    async def get_event_matches(self, event_key: str) -> List[TBAMatchResponse]:
        """
        Fetch all matches for a specific event.

        Args:
            event_key: Event key (e.g., "2023casf")

        Returns:
            List of match responses

        Raises:
            HTTPError: If API request fails
        """
        data = await self._get(f"event/{event_key}/matches")
        return [TBAMatchResponse(**match) for match in data]


class StatboticsAsyncClient:
    """
    Async client for Statbotics API.

    Fetches team performance metrics including EPA, rankings, and records.
    Uses the existing statbotics library but wraps calls in async for
    concurrent execution with configurable limits.
    """

    def __init__(self, max_concurrent: int = 20) -> None:
        """
        Initialize the Statbotics async client.

        Args:
            max_concurrent: Maximum concurrent requests
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._sync_client = Statbotics()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def get_team_year(
        self, team_number: int, year: int
    ) -> Optional[StatboticsTeamMetricsRaw]:
        """
        Fetch metrics for a single team in a given year.

        Args:
            team_number: Team number (e.g., 254)
            year: Year to fetch metrics for

        Returns:
            Team metrics or None if team not found
        """
        async with self.semaphore:
            logger.debug(f"[STATBOTICS] Fetching team {team_number} year {year}")

            # Run blocking statbotics call in thread pool
            loop = asyncio.get_event_loop()
            try:
                logger.debug(f"[STATBOTICS] Calling run_in_executor for team {team_number}")
                data = await loop.run_in_executor(
                    None, self._sync_client.get_team_year, team_number, year
                )
                logger.debug(f"[STATBOTICS] Got data for team {team_number}, validating with Pydantic")
                result = StatboticsTeamMetricsRaw(**data)
                logger.debug(f"[STATBOTICS] Successfully validated team {team_number}")
                return result
            except Exception as e:
                logger.error(f"[STATBOTICS] Failed to fetch team {team_number} year {year}: {type(e).__name__}: {e}")
                return None

    async def get_team_batch(
        self, team_numbers: List[int], year: int, desc: str = "Fetching team metrics"
    ) -> List[Optional[StatboticsTeamMetricsRaw]]:
        """
        Fetch metrics for multiple teams concurrently with progress bar.

        Uses asyncio.gather with progress tracking. Concurrent fetching is
        much faster than sequential while the progress bar updates as tasks complete.

        Args:
            team_numbers: List of team numbers
            year: Year to fetch metrics for
            desc: Description for the progress bar

        Returns:
            List of team metrics (None for failed fetches)
        """
        logger.info(f"[TEAM BATCH] Starting fetch for {len(team_numbers)} teams for year {year}")
        logger.info(f"[TEAM BATCH] First 5 teams: {team_numbers[:5]}")
        logger.info(f"[TEAM BATCH] Last 5 teams: {team_numbers[-5:]}")

        # Create all tasks
        tasks = [self.get_team_year(team, year) for team in team_numbers]

        # Use tqdm to track progress - create completed counter for manual updates
        pbar = tqdm(total=len(tasks), desc=desc, unit="team")
        results = []
        success_count = 0
        failure_count = 0

        # Wrap each coroutine to update progress bar on completion
        async def track_progress(coro):
            """Wrap coroutine to update progress bar on completion."""
            nonlocal success_count, failure_count
            try:
                result = await coro
                if result is not None:
                    success_count += 1
                else:
                    failure_count += 1
                pbar.update(1)
                return result
            except Exception as e:
                failure_count += 1
                pbar.update(1)
                logger.debug(f"[TEAM BATCH] Exception: {e}")
                return None

        # Run all tasks concurrently with progress tracking
        tracked_tasks = [track_progress(task) for task in tasks]
        results = await asyncio.gather(*tracked_tasks)

        pbar.close()

        logger.info(f"[TEAM BATCH] COMPLETE: {success_count} successful, {failure_count} failed out of {len(team_numbers)} total")
        return results


# =============================================================================
# Async Data Scraper - Main scraping class
# =============================================================================


class AsyncDataScraper:
    """
    Async FRC match data scraper.

    Scrapes match data from The Blue Alliance API and team performance metrics
    from Statbotics API. Supports concurrent requests for 10-100x speedup
    compared to sequential scraping.

    Attributes:
        years: List of years to scrape
        events: Optional list of specific event keys to filter (empty = all events)
        teams: Optional list of specific teams to filter (empty = all teams)
        max_concurrent_tba: Max concurrent TBA API requests (default: 50)
        max_concurrent_statbotics: Max concurrent Statbotics requests (default: 20)
        cache_ttl_hours: Cache TTL in hours (default: 24)
    """

    # Default fields to extract from Statbotics metrics
    DEFAULT_FIELDS: List[str] = [
        "team",
        "epa_start",
        "epa_pre_champs",
        "epa_diff",
        "auto_epa_end",
        "teleop_epa_end",
        "endgame_epa_end",
        "rp_1_epa",
        "rp_2_epa",
        "unitless_epa_end",
        "norm_epa_end",
        "epa_conf_1",
        "epa_conf_2",
        "wins",
        "losses",
        "ties",
        "count",
        "winrate",
        "total_epa_rank",
        "total_epa_percentile",
        "country_epa_rank",
        "country_epa_percentile",
        "state_epa_rank",
        "state_epa_percentile",
        "district_epa_rank",
        "district_epa_percentile",
    ]

    def __init__(
        self,
        years: List[int],
        events: List[str],
        teams: List[str],
        max_concurrent_tba: int = 50,
        max_concurrent_statbotics: int = 20,
        cache_ttl_hours: int = 24,
        auto_resume: bool = True,
    ) -> None:
        """
        Initialize the async data scraper.

        Args:
            years: List of years to scrape (e.g., [2023, 2024])
            events: Optional event key filter (e.g., ["2023casf"])
            teams: Optional team filter (e.g., ["frc254"])
            max_concurrent_tba: Max concurrent TBA requests
            max_concurrent_statbotics: Max concurrent Statbotics requests
            cache_ttl_hours: Cache time-to-live in hours
            auto_resume: If True, resume from existing cache files (default: True)
        """
        self.years = years
        self.events = events
        self.teams = teams
        self.max_concurrent_tba = max_concurrent_tba
        self.max_concurrent_statbotics = max_concurrent_statbotics
        self.auto_resume = auto_resume

        # Initialize cache manager
        self.cache = CacheManager(cache_dir="static", ttl_hours=cache_ttl_hours)

        # Get TBA API key from environment
        api_key = os.getenv("X_TBA_AUTH_KEY")
        if not api_key:
            raise ValueError("X_TBA_AUTH_KEY environment variable not set")

        # Initialize API clients (created in context manager)
        self.tba_client: Optional[TBAAsyncClient] = None
        self.statbotics_client = StatboticsAsyncClient(max_concurrent_statbotics)

        # Storage for scraped data
        self.unique_teams: Set[str] = set()
        self.metrics: Dict[int, Dict[str, Any]] = {}

    async def scrape_year_and_prepare_metrics(self, year: int) -> None:
        """
        Scrape all events, matches, and team metrics for a given year.

        This is the main scraping method that:
        1. Checks for existing cache files (if auto_resume is enabled)
        2. Fetches all events for the year (or loads from cache)
        3. Fetches all matches for each event (concurrently)
        4. Collects all unique team numbers
        5. Fetches metrics for all teams (concurrently, skipping cached)
        6. Saves intermediate results to cache

        Performance: ~10-100x faster than sequential due to async concurrency.

        Auto-Resume Behavior:
            - If teams.pk3, events.pk3, matches.pk3 exist and are valid: skips TBA fetching
            - If metrics.pk3 exists and is valid: only fetches missing team metrics
            - Set auto_resume=False to force full re-scrape

        Args:
            year: The year to scrape (e.g., 2023, 2024)

        Raises:
            HTTPError: If API requests fail after retries
        """
        logger.info(f"Starting scrape for year {year}")

        year_dir = Path(f"static/{year}")
        year_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing cache files
        teams_file = year_dir / "teams.pk3"
        events_file = year_dir / "events.pk3"
        matches_file = year_dir / "matches.pk3"
        metrics_file = year_dir / "metrics.pk3"

        # Determine what needs to be scraped
        needs_tba_scrape = not self.auto_resume or not all(
            self.cache.is_valid(f) for f in [teams_file, events_file, matches_file]
        )

        needs_metrics_scrape = not self.auto_resume or not self.cache.is_valid(metrics_file)

        # Step 1 & 2: Fetch TBA data (events/matches) or load from cache
        if needs_tba_scrape:
            logger.info(f"Fetching from TBA API for {year}")
            async with TBAAsyncClient(
                api_key=os.getenv("X_TBA_AUTH_KEY"),
                max_concurrent=self.max_concurrent_tba,
            ) as tba_client:
                self.tba_client = tba_client

                # Fetch all events for the year
                logger.info(f"Fetching events for {year}")
                events = await tba_client.get_events(year)

                # Filter events if specific events were requested
                if self.events:
                    events = [e for e in events if e.key in self.events]

                logger.info(f"Found {len(events)} events for {year}")
                event_keys = [e.key for e in events]

                # Fetch all matches for all events concurrently
                logger.info(f"Fetching matches for {len(events)} events")
                matches_tasks = [tba_client.get_event_matches(key) for key in event_keys]

                # Use tqdm to track event fetching progress
                pbar = tqdm(total=len(matches_tasks), desc=f"Fetching {year} matches", unit="event")
                all_matches_lists = []

                async def track_match_progress(coro):
                    """Wrap coroutine to update progress bar on completion."""
                    result = await coro
                    pbar.update(1)
                    return result

                tracked_tasks = [track_match_progress(task) for task in matches_tasks]
                all_matches_lists = await asyncio.gather(*tracked_tasks)
                pbar.close()

                # Flatten list of lists
                matches: List[TBAMatchResponse] = []
                for match_list in all_matches_lists:
                    matches.extend(match_list)

                logger.info(f"Found {len(matches)} total matches")

                # Collect unique teams
                for match in matches:
                    self.unique_teams.update(match.alliances["red"].team_keys)
                    self.unique_teams.update(match.alliances["blue"].team_keys)

                logger.info(f"Found {len(self.unique_teams)} unique teams")

                # Save teams, events, and matches to cache
                self.cache.save(teams_file, self.unique_teams)
                self.cache.save(events_file, event_keys)

                # Convert matches to dicts for pickling
                matches_dicts = [match.dict() for match in matches]
                self.cache.save(matches_file, matches_dicts)
        else:
            # Load from cache
            logger.info(f"Loading cached TBA data for {year}")
            self.unique_teams = self.cache.load(teams_file)
            event_keys = self.cache.load(events_file)
            matches_dicts = self.cache.load(matches_file)

            logger.info(f"Loaded {len(self.unique_teams)} teams from cache")
            logger.info(f"Loaded {len(matches_dicts)} matches from cache")

        # Step 3: Fetch team metrics (with auto-resume support)
        if needs_metrics_scrape:
            # Load existing metrics if available (for partial resume)
            existing_metrics: Dict[int, Dict[str, Any]] = {}
            if self.cache.is_valid(metrics_file):
                try:
                    existing_metrics = self.cache.load(metrics_file)
                    logger.info(f"Loaded {len(existing_metrics)} cached team metrics")
                except Exception:
                    existing_metrics = {}

            # Determine which teams need metrics fetched
            team_numbers = [
                int(team_key[3:]) for team_key in self.unique_teams if team_key[3:].isdigit()
            ]

            logger.info(f"[SCRAPER] Total unique teams with digits: {len(team_numbers)}")
            logger.info(f"[SCRAPER] Sample teams: {sorted(team_numbers)[:10]}")

            # Filter out teams we already have metrics for
            teams_to_fetch = [t for t in team_numbers if t not in existing_metrics]

            logger.info(f"[SCRAPER] Teams already in cache: {len(existing_metrics)}")
            logger.info(f"[SCRAPER] Teams needing fetch: {len(teams_to_fetch)}")

            if teams_to_fetch:
                logger.info(f"[SCRAPER] Fetching metrics for {len(teams_to_fetch)} teams (skipping {len(team_numbers) - len(teams_to_fetch)} cached)")
                logger.info(f"[SCRAPER] Teams to fetch sample: {sorted(teams_to_fetch)[:10]}")
            else:
                logger.info(f"[SCRAPER] All {len(team_numbers)} team metrics already cached - skipping fetch")

            # Fetch metrics for teams that don't have cached data
            if teams_to_fetch:
                logger.info(f"[SCRAPER] ABOUT TO CALL get_team_batch with {len(teams_to_fetch)} teams")

                metrics_list = await self.statbotics_client.get_team_batch(
                    teams_to_fetch, year, desc=f"Fetching {year} team metrics"
                )

                logger.info(f"[SCRAPER] get_team_batch RETURNED {len(metrics_list)} results")

                # Filter out failed fetches and filter to desired fields
                logger.info(f"[SCRAPER] Starting filter_metrics processing...")
                valid_metrics = []
                for idx, metrics in enumerate(metrics_list):
                    if metrics is not None:
                        try:
                            filtered = self._filter_metrics(metrics)
                            valid_metrics.append(filtered)
                            if (idx + 1) % 100 == 0:
                                logger.info(f"[SCRAPER] Filtered {idx + 1}/{len(metrics_list)} metrics")
                        except Exception as e:
                            logger.error(f"[SCRAPER] Error filtering metrics at index {idx}: {e}")
                    else:
                        logger.debug(f"[SCRAPER] Skipping None metrics at index {idx}")

                logger.info(f"[SCRAPER] Successfully filtered {len(valid_metrics)} valid metrics from {len(metrics_list)} total")

                # Merge with existing metrics
                self.metrics = existing_metrics
                for metric in valid_metrics:
                    team_num = metric["team"]
                    self.metrics[team_num] = {k: v for k, v in metric.items() if k != "team"}

                # Save metrics to cache
                self.cache.save(metrics_file, self.metrics)
            else:
                self.metrics = existing_metrics
        else:
            # Load metrics from cache
            logger.info(f"Loading cached metrics for {year}")
            self.metrics = self.cache.load(metrics_file)
            logger.info(f"Loaded {len(self.metrics)} team metrics from cache")

        logger.info(f"Completed scrape for year {year}")

    async def scrape_all_years(self) -> None:
        """
        Scrape all years specified in the constructor.

        Scrapes each year sequentially but uses concurrent API calls
        within each year for maximum performance.
        """
        for year in self.years:
            await self.scrape_year_and_prepare_metrics(year)

    @staticmethod
    def _filter_metrics(raw_metrics: StatboticsTeamMetricsRaw) -> Dict[str, Any]:
        """
        Filter and transform raw Statbotics metrics to desired fields.

        Extracts generic EPA metrics that apply across all years, ignoring
        year-specific game breakdown data (cubes, cones, links, etc.).

        Args:
            raw_metrics: Raw metrics from Statbotics API

        Returns:
            Filtered metrics dictionary with generic fields only
        """
        filtered: Dict[str, Any] = {}

        # Basic team info
        filtered["team"] = raw_metrics.team

        # Access nested EPA structure
        epa = raw_metrics.epa

        # EPA stats (start, pre_champs from nested stats)
        stats = epa.get("stats", {})
        filtered["epa_start"] = stats.get("start", 0.0)
        filtered["epa_pre_champs"] = stats.get("pre_champs", 0.0)

        # Calculate epa_diff (pre_champs - start)
        filtered["epa_diff"] = filtered["epa_pre_champs"] - filtered["epa_start"]

        # EPA breakdown (generic fields only - skip year-specific cubes/cones/etc)
        breakdown = epa.get("breakdown", {})
        filtered["auto_epa_end"] = breakdown.get("auto_points", 0.0)
        filtered["teleop_epa_end"] = breakdown.get("teleop_points", 0.0)
        filtered["endgame_epa_end"] = breakdown.get("endgame_points", 0.0)
        filtered["rp_1_epa"] = breakdown.get("rp_1", 0.0)
        filtered["rp_2_epa"] = breakdown.get("rp_2", 0.0)

        # EPA values
        filtered["unitless_epa_end"] = epa.get("unitless", 0.0)
        filtered["norm_epa_end"] = epa.get("norm", 0.0)

        # EPA confidence interval
        conf = epa.get("conf", [])
        filtered["epa_conf_1"] = conf[0] if len(conf) > 0 else 0.0
        filtered["epa_conf_2"] = conf[1] if len(conf) > 1 else 0.0

        # Record
        filtered["wins"] = raw_metrics.record.wins
        filtered["losses"] = raw_metrics.record.losses
        filtered["ties"] = raw_metrics.record.ties
        filtered["count"] = raw_metrics.record.count
        filtered["winrate"] = raw_metrics.record.winrate

        # Rankings (from nested epa.ranks structure)
        ranks = epa.get("ranks", {})
        if ranks:
            # Total/Worldwide rank
            total_rank = ranks.get("total", {})
            if isinstance(total_rank, dict):
                filtered["total_epa_rank"] = total_rank.get("rank", 0)
                filtered["total_epa_percentile"] = total_rank.get("percentile", 0.0)

            # Country rank
            country_rank = ranks.get("country")
            if isinstance(country_rank, dict):
                filtered["country_epa_rank"] = country_rank.get("rank")
                filtered["country_epa_percentile"] = country_rank.get("percentile")

            # State rank
            state_rank = ranks.get("state")
            if isinstance(state_rank, dict):
                filtered["state_epa_rank"] = state_rank.get("rank")
                filtered["state_epa_percentile"] = state_rank.get("percentile")

            # District rank
            district_rank = ranks.get("district")
            if isinstance(district_rank, dict):
                filtered["district_epa_rank"] = district_rank.get("rank")
                filtered["district_epa_percentile"] = district_rank.get("percentile")

        # Set default values for missing ranking data
        for field in ["country_epa_rank", "country_epa_percentile",
                      "state_epa_rank", "state_epa_percentile",
                      "district_epa_rank", "district_epa_percentile"]:
            if field not in filtered:
                filtered[field] = None

        return filtered

    async def _format_team_metrics(
        self, team_number: int, placement: str
    ) -> Optional[Dict[str, Any]]:
        """
        Format team metrics for a specific alliance position.

        Creates prefixed keys for team metrics (e.g., "red1_epa_start")
        for use in the final dataset.

        Args:
            team_number: Team number (e.g., 254)
            placement: Alliance position (e.g., "red1", "blue2")

        Returns:
            Dictionary with prefixed metric keys, or None if team not found
        """
        if team_number not in self.metrics:
            return None

        team_data = self.metrics[team_number]
        return {f"{placement}_{key}": value for key, value in team_data.items()}

    async def format_match_with_metrics(
        self, match: TBAMatchResponse, year: int
    ) -> Optional[Dict[str, Any]]:
        """
        Format match data with all 6 teams' metrics.

        Combines basic match information (scores, winner, etc.) with detailed
        metrics for all 6 teams (3 red, 3 blue).

        CRITICAL BUG FIX: The year parameter is now passed explicitly instead
        of using self.years[0]. This ensures matches are labeled with their
        actual year, enabling correct multi-year scraping and time-based
        weighting in dataset.py.

        Args:
            match: Match data from TBA API
            year: Actual year of the match (FIXED - no longer hardcoded)

        Returns:
            Formatted match dictionary with ~171 fields, or None on error
        """
        try:
            # Extract team numbers from alliances
            red_teams = match.alliances["red"].team_keys
            blue_teams = match.alliances["blue"].team_keys

            # Validate we have exactly 3 teams per alliance
            if len(red_teams) < 3 or len(blue_teams) < 3:
                return None

            # Parse team numbers (remove "frc" prefix)
            red1_num = int(red_teams[0][3:]) if red_teams[0][3:].isdigit() else 0
            red2_num = int(red_teams[1][3:]) if red_teams[1][3:].isdigit() else 0
            red3_num = int(red_teams[2][3:]) if red_teams[2][3:].isdigit() else 0
            blue1_num = int(blue_teams[0][3:]) if blue_teams[0][3:].isdigit() else 0
            blue2_num = int(blue_teams[1][3:]) if blue_teams[1][3:].isdigit() else 0
            blue3_num = int(blue_teams[2][3:]) if blue_teams[2][3:].isdigit() else 0

            # Skip match if any team number is invalid
            if 0 in [red1_num, red2_num, red3_num, blue1_num, blue2_num, blue3_num]:
                return None

            # Start with basic match fields
            result: Dict[str, Any] = {
                "match_key": match.key,
                "event_key": match.event_key,
                "match_type": match.comp_level,
                "set_number": match.set_number,
                "match_number": match.match_number,
                # CRITICAL FIX: Use actual year parameter instead of self.years[0]
                "year": year,
                "red1": red1_num,
                "red2": red2_num,
                "red3": red3_num,
                "blue1": blue1_num,
                "blue2": blue2_num,
                "blue3": blue3_num,
                "red_score": match.alliances["red"].score,
                "blue_score": match.alliances["blue"].score,
                "winning_alliance": match.winning_alliance,
            }

            # Add score breakdown if available
            if match.score_breakdown:
                # Get breakdown objects (Pydantic models, not dicts)
                red_breakdown = match.score_breakdown.get("red")
                blue_breakdown = match.score_breakdown.get("blue")

                # Access Pydantic model attributes directly
                result["red_rp"] = red_breakdown.rp if red_breakdown else None
                result["blue_rp"] = blue_breakdown.rp if blue_breakdown else None
                result["auto_points_red"] = red_breakdown.autoPoints if red_breakdown else None
                result["auto_points_blue"] = blue_breakdown.autoPoints if blue_breakdown else None
                result["teleop_points_red"] = red_breakdown.teleopPoints if red_breakdown else None
                result["teleop_points_blue"] = blue_breakdown.teleopPoints if blue_breakdown else None
            else:
                result["red_rp"] = None
                result["blue_rp"] = None
                result["auto_points_red"] = None
                result["auto_points_blue"] = None
                result["teleop_points_red"] = None
                result["teleop_points_blue"] = None

            # Fetch and add metrics for each team
            for team_num, position in [
                (red1_num, "red1"),
                (red2_num, "red2"),
                (red3_num, "red3"),
                (blue1_num, "blue1"),
                (blue2_num, "blue2"),
                (blue3_num, "blue3"),
            ]:
                metrics = await self._format_team_metrics(team_num, position)
                if metrics:
                    result.update(metrics)

            return result

        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Error formatting match {match.key}: {e}")
            return None

    async def build_dataset(self, year: int) -> List[Dict[str, Any]]:
        """
        Build dataset by combining matches with team metrics.

        Loads cached match data and formats each match with corresponding
        team metrics. Skips matches where team metrics are unavailable.

        Args:
            year: Year to build dataset for

        Returns:
            List of formatted match dictionaries ready for CSV export

        Raises:
            FileNotFoundError: If cached matches don't exist
        """
        logger.info(f"Building dataset for year {year}")

        # Load metrics from cache
        year_dir = Path(f"static/{year}")
        self.metrics = self.cache.load(year_dir / "metrics.pk3")
        self.unique_teams = self.cache.load(year_dir / "teams.pk3")

        logger.info(f"Loaded metrics for {len(self.metrics)} teams")

        # Load matches from cache
        matches_dicts = self.cache.load(year_dir / "matches.pk3")

        # Convert back to Pydantic models
        matches = [TBAMatchResponse(**m) for m in matches_dicts]

        logger.info(f"Formatting {len(matches)} matches with team metrics")

        # Format all matches with metrics (with progress bar)
        formatted_matches = []
        for match in tqdm(
            matches,
            desc=f"Formatting {year} matches",
            unit="match",
        ):
            formatted = await self.format_match_with_metrics(match, year)
            if formatted:
                formatted_matches.append(formatted)

        logger.info(f"Successfully formatted {len(formatted_matches)} matches")
        logger.info(f"Completed dataset build for year {year}")

        return formatted_matches

    def load_data_from_files(self, year: int) -> None:
        """
        Load previously scraped data from cache files.

        Useful for reprocessing without rescraping from APIs.

        Args:
            year: Year to load data for
        """
        logger.info(f"Loading cached data for year {year}")

        year_dir = Path(f"static/{year}")
        self.unique_teams = self.cache.load(year_dir / "teams.pk3")
        self.metrics = self.cache.load(year_dir / "metrics.pk3")

        logger.info(f"Loaded {len(self.metrics)} teams and their metrics")


# =============================================================================
# Helper Functions (Preserved for backward compatibility)
# =============================================================================


def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries with intelligent list handling.

    If both values are lists, concatenates them. If one is a list,
    appends the other value. Otherwise combines into a list.

    Args:
        dict1: First dictionary
        dict2: Second dictionary

    Returns:
        Merged dictionary

    Note:
        Only keys present in dict1 are included in the output.
        Keys only in dict2 are not added to the merged result.

    Example:
        >>> d1 = {'teams': ['frc1', 'frc2'], 'score': 100}
        >>> d2 = {'teams': ['frc3'], 'score': 50}
        >>> merge_dictionaries(d1, d2)
        {'teams': ['frc1', 'frc2', 'frc3'], 'score': [100, 50]}
    """
    merged_dict = {}
    for key in dict1:
        # Check if both values are lists
        if isinstance(dict1[key], list) and isinstance(dict2.get(key), list):
            merged_dict[key] = dict1[key] + dict2[key]
        # Check if only dict1's value is a list
        elif isinstance(dict1[key], list):
            merged_dict[key] = dict1[key] + [dict2[key]]
        # Check if only dict2's value is a list
        elif isinstance(dict2.get(key), list):
            merged_dict[key] = [dict1[key]] + dict2[key]
        # If neither value is a list, combine them into a list
        else:
            merged_dict[key] = [dict1[key], dict2.get(key)]
    return merged_dict


# =============================================================================
# Main Execution Block
# =============================================================================


async def main() -> None:
    """
    Main entry point for the scraper.

    Example usage:
        - Scrape single year: AsyncDataScraper([2023], [], [])
        - Scrape multiple years: AsyncDataScraper([2023, 2024], [], [])
        - Scrape specific events: AsyncDataScraper([2023], ["2023casf"], [])
    """
    # Configure which years to scrape
    years_to_scrape = [2023, 2024, 2025]

    # Optional: Filter specific events or teams (empty = all)
    events_filter: List[str] = []
    teams_filter: List[str] = []

    logger.info(f"Starting scraper for years: {years_to_scrape}")

    # Initialize scraper
    scraper = AsyncDataScraper(
        years=years_to_scrape,
        events=events_filter,
        teams=teams_filter,
        max_concurrent_tba=10,  # TBA API can handle many concurrent requests
        max_concurrent_statbotics=10,  # Be more conservative with Statbotics
        cache_ttl_hours=24,  # Cache expires after 24 hours
    )

    # Scrape all years
    await scraper.scrape_all_years()

    # Build dataset and export to CSV for each year
    for year in years_to_scrape:
        logger.info(f"Building dataset for {year}")
        data = await scraper.build_dataset(year)

        # Create DataFrame and export to CSV
        df = pd.DataFrame(data)
        output_file = f"{year}_data.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"Saved {len(data)} matches to {output_file}")
        logger.info(f"Dataset shape: {df.shape}")

    logger.info("Scraping complete!")


if __name__ == "__main__":
    """
    Run the scraper when executed directly.

    Usage:
        python main.py

    The scraper will:
        1. Fetch all events/matches for configured years
        2. Fetch team metrics from Statbotics
        3. Cache results to static/{year}/
        4. Export CSV files for each year
    """
    asyncio.run(main())
