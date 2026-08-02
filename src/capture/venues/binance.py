"""Binance USDs-M perpetual futures. Spot is deliberately not captured - see spec 11 Q4."""
from __future__ import annotations

from capture.venues import ExtractedMeta, StreamSpec

_WS_BASE = "wss://fstream.binance.com/stream?streams="
_INSTRUMENTS_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

_CORE_CHANNELS = ["depth@100ms", "aggTrade", "markPrice@1s", "forceOrder"]
_TAIL_CHANNELS = ["aggTrade", "markPrice@1s", "forceOrder"]

_EVENT_TO_STREAM = {
    "depthUpdate": "depth",
    "aggTrade": "aggTrade",
    "markPriceUpdate": "markPrice",
    "forceOrder": "forceOrder",
}


class BinanceVenue:
    name = "binance"

    def _specs(self, symbols: list[str], channels: list[str]) -> list[StreamSpec]:
        return [
            StreamSpec(self.name, channel.split("@")[0], symbol, f"{symbol.lower()}@{channel}")
            for symbol in symbols
            for channel in channels
        ]

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _CORE_CHANNELS)

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _TAIL_CHANNELS)

    def ws_url(self, specs: list[StreamSpec]) -> str:
        return _WS_BASE + "/".join(spec.channel for spec in specs)

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        return []          # subscription is encoded in the URL

    def extract(self, parsed: dict) -> ExtractedMeta:
        if not isinstance(parsed, dict):
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        body = parsed.get("data", parsed)
        if not isinstance(body, dict):
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        event = body.get("e")
        if not isinstance(event, str):
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        seq = None
        if event == "depthUpdate":
            seq = {k: body[k] for k in ("U", "u", "pu", "T") if k in body} or None

        t_exch_ms = body.get("E")
        if not isinstance(t_exch_ms, int):
            t_exch_ms = None

        symbol = body.get("s")
        if not isinstance(symbol, str) or not symbol:
            symbol = "unknown"

        stream = _EVENT_TO_STREAM.get(event, event)
        return ExtractedMeta(t_exch_ms, seq, "data", stream, symbol)

    def instruments_url(self) -> str:
        return _INSTRUMENTS_URL

    def parse_instruments(self, payload: dict) -> list[str]:
        if not isinstance(payload, dict):
            return []
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            return []

        result = []
        for item in symbols:
            if not isinstance(item, dict):
                continue
            if item.get("contractType") != "PERPETUAL" or item.get("status") != "TRADING":
                continue
            symbol = item.get("symbol")
            if isinstance(symbol, str) and symbol:
                result.append(symbol)
        return sorted(result)
