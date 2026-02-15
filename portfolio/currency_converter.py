"""
FX rate conversion using instrument prices as implied exchange rates.
"""
from typing import Dict, Optional


class CurrencyConverter:
    """
    Converts amounts between currencies using current market prices.
    
    Example: 
      To convert EUR to USD when EUR_USD = 1.0800:
        convert(1000, "EUR", "USD") = 1000 * 1.0800 = 1080.00 USD
    """
    
    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        rates: Dict[str, float]
    ) -> float:
        if from_currency == to_currency:
            return amount
        
        # Direct pair (e.g., EUR_USD for EUR→USD)
        direct_pair = f"{from_currency}_{to_currency}"
        if direct_pair in rates:
            return amount * rates[direct_pair]
        
        # Inverse pair (e.g., USD_EUR for USD→EUR when we have EUR_USD)
        inverse_pair = f"{to_currency}_{from_currency}"
        if inverse_pair in rates:
            return amount / rates[inverse_pair]
        
        # Cross-currency via USD (e.g., EUR→JPY via EUR_USD and USD_JPY)
        if from_currency != "USD" and to_currency != "USD":
            usd_rate_from = self._get_usd_rate(from_currency, rates)
            usd_rate_to = self._get_usd_rate(to_currency, rates)
            if usd_rate_from and usd_rate_to:
                usd_amount = amount * usd_rate_from
                return usd_amount * usd_rate_to
        
        raise ValueError(
            f"Cannot convert {from_currency} to {to_currency} - missing rate. "
            f"Available rates: {list(rates.keys())}"
        )
    
    def _get_usd_rate(self, currency: str, rates: Dict[str, float]) -> Optional[float]:
        """Get rate to convert currency to USD"""
        if currency == "USD":
            return 1.0
        
        # Direct USD pair
        if f"{currency}_USD" in rates:
            return rates[f"{currency}_USD"]
        if f"USD_{currency}" in rates:
            return 1.0 / rates[f"USD_{currency}"]
        
        return None