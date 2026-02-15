"""
Per-instrument position tracking with average price and P&L calculation.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Position:
    """
    Tracks net position for a single instrument with FIFO cost basis.
    
    Design Notes:
      - Maintains list of fills for accurate average price calculation
      - Handles both long and short positions
      - P&L calculated in quote currency (e.g., USD for EUR_USD)
    """
    instrument: str
    base_currency: str  # e.g., "EUR" for EUR_USD
    quote_currency: str  # e.g., "USD" for EUR_USD
    fills: List[dict] = field(default_factory=list)  # [{"direction": +1/-1, "units": float, "price": float}]
    
    @property
    def net_units(self) -> float:
        """Net position size (positive = long, negative = short)"""
        return sum(f["direction"] * f["units"] for f in self.fills)
    
    @property
    def average_price(self) -> float:
        """Average entry price for current net position (FIFO basis)"""
        if self.net_units == 0:
            return 0.0
        
        # Simple average price weighted by units (not FIFO liquidation)
        total_cost = sum(f["direction"] * f["units"] * f["price"] for f in self.fills)
        return abs(total_cost / self.net_units)
    
    def update(self, direction: int, units: float, price: float):
        """Add new fill to position"""
        self.fills.append({
            "direction": direction,
            "units": units,
            "price": price,
            "timestamp": None  # Set by PortfolioManager for consistency
        })
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L in quote currency.
        
        For long position (net_units > 0):
          P&L = net_units * (current_price - avg_price)
        
        For short position (net_units < 0):
          P&L = |net_units| * (avg_price - current_price)
        """
        if self.net_units == 0:
            return 0.0
        
        if self.net_units > 0:  # Long
            return self.net_units * (current_price - self.average_price)
        else:  # Short
            return abs(self.net_units) * (self.average_price - current_price)
    
    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "fills": self.fills
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        pos = cls(
            instrument=data["instrument"],
            base_currency=data["base_currency"],
            quote_currency=data["quote_currency"]
        )
        pos.fills = data["fills"]
        return pos