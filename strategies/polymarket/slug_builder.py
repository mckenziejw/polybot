"""Event slug builder for Polymarket BTC UpDown 5-minute markets.

Generates event slugs for the current and upcoming 5-minute windows.
Plugs into NautilusTrader's PolymarketInstrumentProviderConfig via:

    PolymarketInstrumentProviderConfig(
        event_slug_builder="strategies.polymarket.slug_builder:build_btc_5m_slugs",
    )
"""

import time


def build_btc_5m_slugs() -> list[str]:
    """Return event slugs for the previous, current, and next two 5-min BTC UpDown markets.

    Polymarket uses the pattern: btc-updown-5m-{aligned_unix_seconds}
    where the timestamp is floor-aligned to 300-second boundaries.
    """
    now = int(time.time())
    aligned = (now // 300) * 300
    return [f"btc-updown-5m-{aligned + i * 300}" for i in range(-1, 3)]
