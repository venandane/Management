"""
Multi-currency trading account with position tracking and P&L calculation.
Designed for cross-account trading scenarios (e.g., buy from Account A, sell from Account B).
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional
import uuid
import logging
from trading_system.portfolio.position import Position

logger = logging.getLogger(__name__)


@dataclass
class Account:
    """
    Represents a trading account with its own currency balance and positions.
    
    Design Philosophy:
      - Each account maintains INDEPENDENT cash balance in its base currency
      - Positions tracked per instrument (e.g., EUR_USD position in Account A ≠ Account B)
      - No automatic cross-account netting (explicit transfers required)
      - All timestamps in UTC for audit consistency
    """
    account_id: str
    account_name: str
    base_currency: str  # e.g., "USD", "EUR"
    initial_balance: float
    current_balance: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        self.current_balance = self.initial_balance
        self._validate_currency()
    
    def _validate_currency(self):
        """Validate base currency format (ISO 4217)"""
        if len(self.base_currency) != 3 or not self.base_currency.isalpha():
            raise ValueError(f"Invalid base currency: {self.base_currency}")
    
    def deposit(self, amount: float, reason: str = "deposit") -> str:
        """Add funds to account"""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self.current_balance += amount
        tx_id = self._log_transaction("DEPOSIT", amount, reason)
        logger.info(f"Account {self.account_id}: +{amount:.2f} {self.base_currency} ({reason})")
        return tx_id
    
    def withdraw(self, amount: float, reason: str = "withdrawal") -> str:
        """Remove funds from account (must have sufficient balance"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.current_balance:
            raise ValueError(f"Insufficient balance: {self.current_balance:.2f} < {amount:.2f}")
        
        self.current_balance -= amount
        tx_id = self._log_transaction("WITHDRAWAL", -amount, reason)
        logger.info(f"Account {self.account_id}: -{amount:.2f} {self.base_currency} ({reason})")
        return tx_id
    
    def execute_trade(
        self,
        instrument: str,
        direction: int,  # +1 = buy/long, -1 = sell/short
        units: float,
        price: float,
        commission: float = 0.0,
        account_counterparty: Optional[str] = None
    ) -> str:
        """
        Execute trade against this account's balance.
        
        Critical Design for Cross-Account Trading:
          - BUY (direction=+1): Deduct quote currency (e.g., USD for EUR_USD)
          - SELL (direction=-1): Add quote currency
          - Position tracking: Base currency exposure (e.g., EUR for EUR_USD)
        
        Example EUR_USD trade in USD account:
          - Buy 10k EUR_USD @ 1.0800:
            * Deduct: 10,000 * 1.0800 = $10,800 USD (quote currency)
            * Add position: +10,000 EUR (base currency exposure)
        """
        if units <= 0:
            raise ValueError("Trade units must be positive")
        
        # Parse instrument to get base/quote currencies
        base_ccy, quote_ccy = self._parse_instrument(instrument)
        
        # Validate currency alignment
        if quote_ccy != self.base_currency:
            raise ValueError(
                f"Account currency mismatch: Account holds {self.base_currency}, "
                f"but trade requires {quote_ccy} for {instrument}"
            )
        
        # Calculate cash impact (always in account's base currency = quote currency)
        cash_impact = -direction * units * price  # Buy: negative cash flow, Sell: positive
        
        # Apply commission (always deducted from account)
        total_cost = abs(cash_impact) + commission
        
        # Check balance for buys (direction=+1 requires cash outflow)
        if direction > 0 and total_cost > self.current_balance:
            raise ValueError(
                f"Insufficient balance for buy: {self.current_balance:.2f} {self.base_currency} "
                f"< {total_cost:.2f} required"
            )
        
        # Update cash balance (commission always paid by executing account)
        self.current_balance += cash_impact - commission
        
        # Update position
        if instrument not in self.positions:
            self.positions[instrument] = Position(
                instrument=instrument,
                base_currency=base_ccy,
                quote_currency=quote_ccy
            )
        
        self.positions[instrument].update(direction, units, price)
        
        # Log transaction
        tx_id = self._log_transaction(
            "BUY" if direction > 0 else "SELL",
            cash_impact,
            f"{instrument} {abs(units):,.0f} @ {price:.5f} (comm: {commission:.2f})",
            metadata={
                "instrument": instrument,
                "direction": direction,
                "units": units,
                "price": price,
                "commission": commission,
                "base_ccy": base_ccy,
                "quote_ccy": quote_ccy,
                "counterparty_account": account_counterparty
            }
        )
        
        logger.info(
            f"Account {self.account_id}: {'BUY' if direction > 0 else 'SELL'} "
            f"{abs(units):,.0f} {instrument} @ {price:.5f} "
            f"| Cash flow: {cash_impact:+.2f} {self.base_currency} "
            f"| Commission: -{commission:.2f} "
            f"| Balance: {self.current_balance:.2f}"
        )
        
        return tx_id
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L across all positions using current market prices"""
        pnl = 0.0
        for instrument, position in self.positions.items():
            if instrument in current_prices and position.net_units != 0:
                pnl += position.calculate_unrealized_pnl(current_prices[instrument])
        return pnl
    
    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """Total equity = cash balance + unrealized P&L"""
        return self.current_balance + self.get_unrealized_pnl(current_prices)
    
    def _parse_instrument(self, instrument: str) -> tuple[str, str]:
        """Parse OANDA-style instrument (e.g., 'EUR_USD' → ('EUR', 'USD'))"""
        parts = instrument.split('_')
        if len(parts) != 2 or len(parts[0]) != 3 or len(parts[1]) != 3:
            raise ValueError(f"Invalid instrument format: {instrument}")
        return parts[0], parts[1]
    
    def _log_transaction(self, tx_type: str, amount: float, description: str, meta dict = None) -> str:
        """Generate transaction ID (actual persistence handled by TransactionLog)"""
        tx_id = f"TX-{self.account_id}-{uuid.uuid4().hex[:8]}"
        return tx_id
    
    def to_dict(self) -> dict:
        """Serialize account state for persistence"""
        return {
            "account_id": self.account_id,
            "account_name": self.account_name,
            "base_currency": self.base_currency,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "positions": {instr: pos.to_dict() for instr, pos in self.positions.items()},
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Account':
        """Deserialize account state"""
        account = cls(
            account_id=data["account_id"],
            account_name=data["account_name"],
            base_currency=data["base_currency"],
            initial_balance=data["initial_balance"]
        )
        
        # Set fields initialized in __post_init__
        account.current_balance = data["current_balance"]
        account.created_at = datetime.fromisoformat(data["created_at"])
        
        # Deserialize positions (critical fix: set on 'account' instance, not 'self')
        account.positions = {
            instr: Position.from_dict(pos_data) 
            for instr, pos_data in data.get("positions", {}).items()
        }
        
        return account