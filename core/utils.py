"""
Core utilities for forex-specific calculations.
"""
from typing import Dict

# Standard pip sizes for major FX pairs
PIP_SIZES: Dict[str, float] = {
    # Major pairs (non-JPY)
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "AUD_USD": 0.0001,
    "NZD_USD": 0.0001,
    "USD_CAD": 0.0001,
    "USD_CHF": 0.0001,
    # JPY pairs (2 decimal places)
    "USD_JPY": 0.01,
    "EUR_JPY": 0.01,
    "GBP_JPY": 0.01,
    "AUD_JPY": 0.01,
    # Gold/Silver
    "XAU_USD": 0.01,   # Gold typically quoted to 2 decimals
    "XAG_USD": 0.001,  # Silver to 3 decimals
}

def get_pip_size(instrument: str) -> float:
    """
    Get pip size for instrument. Handles common naming variations.
    
    Args:
        instrument: OANDA-style instrument name (e.g., "EUR_USD")
    
    Returns:
        Pip size (e.g., 0.0001 for EUR_USD, 0.01 for USD_JPY)
    
    Raises:
        ValueError: If instrument not recognized
    """
    # Normalize instrument name (handle variations)
    norm_instr = instrument.upper().replace("/", "_").replace("-", "_")
    
    # Try exact match first
    if norm_instr in PIP_SIZES:
        return PIP_SIZES[norm_instr]
    
    # Handle cross pairs by checking components
    if "_JPY" in norm_instr:
        return 0.01
    
    # Default to standard 0.0001 for non-JPY pairs
    if any(curr in norm_instr for curr in ["EUR", "GBP", "AUD", "NZD", "CAD", "CHF"]):
        return 0.0001
    
    raise ValueError(f"Unknown instrument: {instrument}. Please add to PIP_SIZES in core/utils.py")

def calculate_pip_distance(price1: float, price2: float, instrument: str) -> float:
    """
    Calculate distance between two prices in pips.
    
    Args:
        price1: First price
        price2: Second price
        instrument: Instrument name
    
    Returns:
        Absolute distance in pips
    """
    pip_size = get_pip_size(instrument)
    return abs(price1 - price2) / pip_size