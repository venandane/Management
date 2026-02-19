#!/usr/bin/env python3
"""
Thin orchestrator for Strong Candle Close Refined strategy analysis.
Delegates all logic to strategy class - script handles only I/O and plotting.
"""
import yaml
import pandas as pd
from pathlib import Path
from trading_system.data.processing.candle_manager import CandleManager
from trading_system.data.processing.resampler import ConfigDrivenResampler
from trading_system.visualization.candle_plotter import CandlePlotter
from trading_system.strategies.registry import StrategyRegistry
from trading_system.core.logger import get_logger
from trading_system.utils.project_paths import CONFIG_DIR
from trading_system.utils.project_paths import DATA_DIR
logger = get_logger(__name__)  # Automatically creates logs/strategies/strong_candle_close_refined.log

print(logger)



def load_config(config_name: str = "strategies.yaml") -> dict:
    config_path = CONFIG_DIR / config_name
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Load config and parameters
    config = load_config()
    instrument = "EUR_USD"
    strategy_params = {
        'min_body_pips': 5.0,
        'min_range_pips': 3.0,
        'adverse_move_pips': 12.0,
        'profit_target_pips': 12.0,
        'bad_signal_lookback_days': 30,
        'tolerance_pips': 0.5
    }

    # Initialize candle manager
    data_root_path = DATA_DIR / "acquisition" / "data" / "raw" / "oanda"
    candle_mgr = CandleManager(data_root_path)

    ConfigDrivenResampler.resample_all_required(
        candle_mgr=candle_mgr,
        instruments=[instrument],
        timeframes_by_instrument={instrument: {"M1", "H1"}},
        base_granularity="M1"
    )
    
    h1_candles = candle_mgr.get_candles(instrument, "H1")
    m1_candles = candle_mgr.get_candles(instrument, "M1")
    h1_candles.attrs['timeframe'] = "H1"
    
    # DELEGATE ALL LOGIC TO STRATEGY CLASS
    logger.info("Initializing strategy...")
    strategy = StrategyRegistry.create(
        name="STRONG_CANDLE_CLOSE_REFINED",
        symbol=instrument,
        timeframes=["M1", "H1"],
        params=strategy_params
    )
    
    logger.info("Running analysis...")
    all_signals, accepted_signals, rejected_signals, bad_signals = strategy.analyze(
        h1_candles=h1_candles,
        m1_candles=m1_candles
    )
    
    # Initialize ALL columns required by plotter's _add_strong_candle_markers()
    enriched_candles = h1_candles.copy()    
    enriched_candles['strong_bull'] = False
    enriched_candles['strong_bear'] = False
    enriched_candles['body_pips'] = 0.0
    enriched_candles['upper_tail_ratio'] = 0.0
    enriched_candles['lower_tail_ratio'] = 0.0
    enriched_candles['range_pips'] = 0.0
    for idx, row in all_signals.iterrows():
        if idx in enriched_candles.index:
            enriched_candles.at[idx, 'strong_bull'] = (row['direction'] > 0)
            enriched_candles.at[idx, 'strong_bear'] = (row['direction'] < 0)
            enriched_candles.at[idx, 'body_pips'] = row['body_pips']
            # Approximate range_pips (body * 1.5) since we don't store exact range in signals
            enriched_candles.at[idx, 'range_pips'] = abs(row['body_pips']) * 1.5
            # Tail ratios not critical for visualization - set to 0.0
            enriched_candles.at[idx, 'upper_tail_ratio'] = 0.0
            enriched_candles.at[idx, 'lower_tail_ratio'] = 0.0
    enriched_candles.attrs['timeframe'] = "H1"
    
    # Format bad signals for plotter
    bad_for_plot = None
    if not bad_signals.empty:
        bad_for_plot = bad_signals[['candle_low', 'candle_high', 'direction']].copy()
        bad_for_plot['price'] = bad_for_plot.apply(
            lambda r: r['candle_low'] if r['direction'] < 0 else r['candle_high'],
            axis=1
        )
    
    # Plot results
    plotter = CandlePlotter(theme="dark", timezone="US/Eastern")
    fig = plotter.plot_strong_candle_refined_v3(
        instrument=instrument,
        candles=enriched_candles,
        accepted_signals=accepted_signals,
        rejected_signals=rejected_signals,
        bad_signals=bad_for_plot,
        title_suffix="(Refined Acceptance Rules)",
        width=3600,
        height=1900
    )
    
    # Save
    data_output_path = DATA_DIR / "output" / "strategy_refined"
    output_dir = Path(data_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{instrument}_H1_refined.html"
    plotter.save_html(fig, str(filepath))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"Strong candles: {len(all_signals):,}")
    logger.info(f"Bad signals:    {len(bad_signals):,}")
    logger.info(f"Accepted:       {len(accepted_signals):,}")
    logger.info(f"Rejected:       {len(rejected_signals):,}")
    logger.info(f"\n✅ Plot saved to: {filepath.absolute()}")
    logger.debug("="*70)


if __name__ == "__main__":
    main()