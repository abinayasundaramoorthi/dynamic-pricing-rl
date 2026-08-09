"""
currency_converter.py

Converts the agent's price recommendations into other currencies, AND
optionally adjusts the price itself based on location - not just the
currency it's displayed in.

Why adjust the price, not just the currency?
Simply converting $200 USD to INR at the market exchange rate gives a
mathematically correct but often commercially wrong number - the same
absolute price rarely makes sense across very different markets. Real
businesses commonly apply LOCATION-BASED PRICE ADJUSTMENTS on top of
currency conversion (purchasing power differences, local competition,
willingness to pay) before converting - e.g. a hotel room might be
deliberately priced lower in one market and higher in another, not just
displayed in a different currency at the same underlying value.

This module supports both:
  - Currency-only conversion (CurrencyConverter) - same price, new currency
  - Location-aware pricing (LocationPricingEngine) - adjusted price, THEN
    converted to that location's currency

Usage
-----
    from evaluation.currency_converter import CurrencyConverter, LocationPricingEngine

    # Currency-only (unchanged from before)
    converter = CurrencyConverter()
    print(converter.format_price(200, "USD", "INR"))

    # Location-aware (NEW): adjusts price AND converts currency
    engine = LocationPricingEngine()
    result = engine.get_localized_price(200, location="IN")
    print(result)
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional


# Static exchange rate table: how many units of each currency equal 1 USD.
# NOTE: illustrative, fixed rates for demo/offline use - not live market
# rates. Update periodically, or supply a `rate_provider` callable for
# live rates.
DEFAULT_RATES_PER_USD: Dict[str, float] = {
    "USD": 1.0,
    "INR": 83.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.5,
    "AUD": 1.52,
    "CAD": 1.36,
    "AED": 3.67,
}

CURRENCY_SYMBOLS: Dict[str, str] = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "AED": "AED ",
}


@dataclass(frozen=True)
class LocationConfig:
    """
    Configuration for one location: which currency it displays prices in,
    and how much to adjust the underlying price before converting.

    price_multiplier : float
        Applied to the BASE (USD) price before currency conversion.
        1.0 = no adjustment (price is identical in underlying value,
        just shown in local currency). Below 1.0 = cheaper in this
        market (e.g. lower cost-of-living / purchasing power / more
        local competition). Above 1.0 = more expensive in this market
        (e.g. premium/high-demand market).
    """
    currency: str
    price_multiplier: float
    display_name: str


# Illustrative starting values - NOT based on real purchasing-power-parity
# data. Intended as a configurable starting point; replace with real
# market research / PPP index data before using in production.
DEFAULT_LOCATION_CONFIG: Dict[str, LocationConfig] = {
    "US": LocationConfig(currency="USD", price_multiplier=1.00, display_name="United States"),
    "IN": LocationConfig(currency="INR", price_multiplier=0.55, display_name="India"),
    "GB": LocationConfig(currency="GBP", price_multiplier=0.95, display_name="United Kingdom"),
    "DE": LocationConfig(currency="EUR", price_multiplier=0.90, display_name="Germany"),
    "JP": LocationConfig(currency="JPY", price_multiplier=0.85, display_name="Japan"),
    "AU": LocationConfig(currency="AUD", price_multiplier=1.05, display_name="Australia"),
    "CA": LocationConfig(currency="CAD", price_multiplier=0.98, display_name="Canada"),
    "AE": LocationConfig(currency="AED", price_multiplier=1.10, display_name="United Arab Emirates"),
}


class CurrencyConverter:
    """Currency-only conversion - unchanged behavior from before this feature was added."""

    def __init__(
        self,
        rates_per_usd: Optional[Dict[str, float]] = None,
        rate_provider: Optional[Callable[[str, str], float]] = None,
    ):
        self.rates_per_usd = rates_per_usd or DEFAULT_RATES_PER_USD
        self.rate_provider = rate_provider

    def get_rate(self, from_currency: str, to_currency: str) -> float:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()

        if self.rate_provider is not None:
            return self.rate_provider(from_currency, to_currency)

        if from_currency not in self.rates_per_usd:
            raise ValueError(f"Unknown currency code: '{from_currency}'. "
                              f"Known currencies: {list(self.rates_per_usd.keys())}")
        if to_currency not in self.rates_per_usd:
            raise ValueError(f"Unknown currency code: '{to_currency}'. "
                              f"Known currencies: {list(self.rates_per_usd.keys())}")

        rate_from_to_usd = 1.0 / self.rates_per_usd[from_currency]
        rate_usd_to_to = self.rates_per_usd[to_currency]
        return rate_from_to_usd * rate_usd_to_to

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        rate = self.get_rate(from_currency, to_currency)
        return round(amount * rate, 2)

    def format_price(self, amount: float, from_currency: str, to_currency: str) -> str:
        converted = self.convert(amount, from_currency, to_currency)
        symbol = CURRENCY_SYMBOLS.get(to_currency.upper(), to_currency.upper() + " ")
        return f"{symbol}{converted:,.2f}"

    def convert_to_all(self, amount: float, from_currency: str) -> Dict[str, str]:
        return {
            currency: self.format_price(amount, from_currency, currency)
            for currency in self.rates_per_usd
        }


@dataclass
class LocalizedPrice:
    """Result of a location-aware pricing calculation."""
    location_code: str
    location_name: str
    base_price_usd: float
    price_multiplier: float
    adjusted_price_usd: float   # base price after the location's multiplier, still in USD
    currency: str
    localized_price: float      # adjusted price converted into the location's currency
    formatted: str               # ready-to-display string, e.g. "₹9,130.00"


class LocationPricingEngine:
    """
    Combines a per-location price adjustment with currency conversion, so
    the SAME base recommendation from the pricing agent can be shown as a
    genuinely different (not just re-labeled) price per market.
    """

    def __init__(
        self,
        location_config: Optional[Dict[str, LocationConfig]] = None,
        converter: Optional[CurrencyConverter] = None,
    ):
        self.location_config = location_config or DEFAULT_LOCATION_CONFIG
        self.converter = converter or CurrencyConverter()

    def get_localized_price(self, base_price_usd: float, location: str) -> LocalizedPrice:
        """
        Parameters
        ----------
        base_price_usd : float
            The agent's recommended price, in USD (this project's base currency).
        location : str
            A location code, e.g. "IN", "US", "GB" - see DEFAULT_LOCATION_CONFIG
            for supported codes.

        Returns
        -------
        LocalizedPrice
        """
        location = location.upper()
        if location not in self.location_config:
            raise ValueError(
                f"Unknown location code: '{location}'. "
                f"Known locations: {list(self.location_config.keys())}"
            )

        config = self.location_config[location]
        adjusted_price_usd = round(base_price_usd * config.price_multiplier, 2)
        localized_price = self.converter.convert(adjusted_price_usd, "USD", config.currency)
        formatted = self.converter.format_price(adjusted_price_usd, "USD", config.currency)

        return LocalizedPrice(
            location_code=location,
            location_name=config.display_name,
            base_price_usd=base_price_usd,
            price_multiplier=config.price_multiplier,
            adjusted_price_usd=adjusted_price_usd,
            currency=config.currency,
            localized_price=localized_price,
            formatted=formatted,
        )

    def get_localized_prices_for_all(self, base_price_usd: float) -> Dict[str, LocalizedPrice]:
        """Computes the localized price for every configured location at once - useful for a dashboard table."""
        return {
            code: self.get_localized_price(base_price_usd, code)
            for code in self.location_config
        }


if __name__ == "__main__":
    converter = CurrencyConverter()
    engine = LocationPricingEngine()

    recommended_price_usd = 200.0

    print(f"Recommended price (agent output, USD base): ${recommended_price_usd:.2f}\n")

    print("=" * 65)
    print("CURRENCY-ONLY CONVERSION (same price, different display currency)")
    print("=" * 65)
    for currency, formatted in converter.convert_to_all(recommended_price_usd, "USD").items():
        print(f"  {currency}: {formatted}")

    print()
    print("=" * 65)
    print("LOCATION-AWARE PRICING (adjusted price + local currency)")
    print("=" * 65)
    for code, result in engine.get_localized_prices_for_all(recommended_price_usd).items():
        print(f"  {result.location_name:<20} ({code}): "
              f"multiplier={result.price_multiplier:.2f} -> "
              f"${result.adjusted_price_usd:.2f} USD equiv -> {result.formatted}")

    print()
    print("Direct lookup example:")
    india_result = engine.get_localized_price(200, "IN")
    print(f"  {india_result}")