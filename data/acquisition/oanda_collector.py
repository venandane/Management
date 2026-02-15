import os
import pickle
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import yaml
import logging
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.instruments import InstrumentsCandles
import oandapyV20.endpoints.instruments as instruments
from trading_system.utils.project_paths import CONFIG_DIR
from trading_system.utils.project_paths import ACQUISITION_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('oanda_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OandaDataCollector:
    """Production-ready OANDA 1-minute candle collector with resume capability"""
    
    MAX_CANDLES_PER_REQUEST = 5000  # OANDA hard limit
    
    def __init__(self, config_name: str = "oanda.yaml"):
        config_path = CONFIG_DIR / config_name
        self.config = self._load_config(config_path)
        self.api = self._init_api()
        self.output_root = ACQUISITION_DIR / Path(self.config['data_collection']['output_path'])
        self.output_root.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_api(self) -> API:
        env = self.config['oanda']['environment']
        token = self.config['oanda']['api_token']
        
        if env == "practice":
            api_url = "https://api-fxpractice.oanda.com"
        elif env == "live":
            api_url = "https://api-fxtrade.oanda.com"
        else:
            raise ValueError(f"Unknown environment: {env}")
        
        return API(access_token=token, environment=env)
    
    def collect_all_instruments(self) -> Dict[str, pd.DataFrame]:
        """Collect data for all configured instruments"""
        results = {}
        instruments = self.config['data_collection']['instruments']
        
        for i, instrument in enumerate(instruments, 1):
            logger.info(f"[{i}/{len(instruments)}] Collecting {instrument}...")
            try:
                df = self.collect_instrument(
                    instrument=instrument,
                    granularity=self.config['data_collection']['granularity'],
                    lookback_days=self.config['data_collection']['lookback_days']
                )
                results[instrument] = df
                logger.info(f"Collected {len(df):,} candles for {instrument}")
            except Exception as e:
                logger.error(f"Failed to collect {instrument}: {str(e)}")
                continue
        
        return results
    
    def collect_instrument(
        self,
        instrument: str,
        granularity: str = "M1",
        lookback_days: int = 430,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Collect candles with automatic pagination handling.
        Resumes from last saved checkpoint if partial data exists.
        """
        # Determine time range
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(days=lookback_days)
        
        # Check for existing partial data to enable resume
        existing_df = self._load_existing_data(instrument, granularity)
        if existing_df is not None and not existing_df.empty:
            last_timestamp = existing_df.index.max()
            if last_timestamp > start_time:
                logger.info(f"Resuming collection from {last_timestamp} (skipping {len(existing_df):,} existing candles)")
                start_time = last_timestamp + timedelta(minutes=1)
        
        # Collect new data in paginated batches
        all_candles = []
        current_start = start_time
        
        while current_start < end_time:
            batch_end = min(
                current_start + timedelta(minutes=self.config['data_collection']['batch_size']),
                end_time
            )
            
            try:
                candles = self._fetch_candle_batch(
                    instrument=instrument,
                    granularity=granularity,
                    from_time=current_start,
                    to_time=batch_end
                )
                
                if not candles:
                    logger.warning(f"No candles returned for {instrument} {current_start} → {batch_end}")
                    current_start = batch_end
                    continue
                
                all_candles.extend(candles)
                current_start = self._parse_oanda_time(candles[-1]['time']) + timedelta(minutes=1)
                
                # Progress logging every 10 batches
                if len(all_candles) % (self.config['data_collection']['batch_size'] * 10) == 0:
                    logger.info(f"Collected {len(all_candles):,} candles so far...")
                
                # Respect rate limits
                time.sleep(self.config['data_collection']['rate_limit_delay'])
                
            except V20Error as e:
                if "rate limit" in str(e).lower():
                    logger.warning("Rate limited - sleeping 5 seconds")
                    time.sleep(5)
                    continue
                raise
        
        # Convert to DataFrame and merge with existing data
        new_df = self._candles_to_dataframe(all_candles, instrument, granularity)
        final_df = self._merge_with_existing(new_df, existing_df)
        
        # Save to disk
        self._save_to_pickle(final_df, instrument, granularity)
        
        return final_df
    
    def _fetch_candle_batch(
        self,
        instrument: str,
        granularity: str,
        from_time: datetime,
        to_time: datetime
    ) -> List[dict]:
        """Fetch a single batch of candles handling OANDA's pagination"""
        params = {
            "granularity": granularity,
            "from": from_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": to_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "includeFirst": True,
            "includeLast": False
        }
        
        r = InstrumentsCandles(instrument=instrument, params=params)
        try:
            self.api.request(r)
            return r.response.get('candles', [])
        except V20Error as e:
            logger.error(f"OANDA API error ({instrument} {from_time}→{to_time}): {e}")
            raise
    
    def _candles_to_dataframe(
        self,
        candles: List[dict],
        instrument: str,
        granularity: str
    ) -> pd.DataFrame:
        """Convert OANDA candle response to pandas DataFrame"""
        if not candles:
            return pd.DataFrame()
        
        records = []
        for c in candles:
            if not c['complete']:  # Skip incomplete (current) candle
                continue
                
            records.append({
                'timestamp': self._parse_oanda_time(c['time']),
                'open': float(c['mid']['o']),
                'high': float(c['mid']['h']),
                'low': float(c['mid']['l']),
                'close': float(c['mid']['c']),
                'volume': int(c['volume']),
                'instrument': instrument,
                'granularity': granularity
            })
        
        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    @staticmethod
    def _parse_oanda_time(time_str: str) -> datetime:
        """Parse OANDA's RFC3339 timestamp to timezone-aware datetime"""
        # Handle OANDA's format: "2024-01-15T12:34:56.789000000Z"
        if '.' in time_str:
            time_str = time_str.split('.')[0] + 'Z'
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    
    def _load_existing_data(
        self,
        instrument: str,
        granularity: str
    ) -> Optional[pd.DataFrame]:
        """Load existing pickle file if it exists for resume capability"""
        filepath = self._get_pickle_path(instrument, granularity)
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    return data['df']
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")
        return None
    
    def _merge_with_existing(
        self,
        new_df: pd.DataFrame,
        existing_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge new data with existing data, removing duplicates"""
        if existing_df is None or existing_df.empty:
            return new_df
        
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]  # Remove duplicates
        combined.sort_index(inplace=True)
        return combined
    
    def _save_to_pickle(
        self,
        df: pd.DataFrame,
        instrument: str,
        granularity: str
    ):
        """Save DataFrame with metadata to pickle file"""
        filepath = self._get_pickle_path(instrument, granularity)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'instrument': instrument,
            'granularity': granularity,
            'start_time': df.index.min(),
            'end_time': df.index.max(),
            'count': len(df),
            'downloaded_at': datetime.now(timezone.utc),
            'source': 'oanda_v20_api'
        }
        
        data = {
            'df': df,
            'metadata': metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {len(df):,} candles to {filepath}")
    
    def _get_pickle_path(self, instrument: str, granularity: str) -> Path:
        """Generate standardized pickle path"""
        return self.output_root / instrument / granularity / f"{instrument}_{granularity}.pkl"


# Usage example
if __name__ == "__main__":
    collector = OandaDataCollector()
    
    try:
        results = collector.collect_all_instruments()
        
        # Quick validation report
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        for instrument, df in results.items():
            print(f"{instrument:12s} | {len(df):>7,} candles | {df.index.min()} → {df.index.max()}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user - data saved up to current point")
    except Exception as e:
        logger.exception(f"Critical failure: {e}")