"""CDAN-TS baseline alias for the time-series CDAN implementation."""

from __future__ import annotations

from .cdan import CDANMethod


class CDANTSMethod(CDANMethod):
    """CDAN configured and reported as the time-series baseline CDAN-TS."""

    method_name = "cdan_ts"
