"""Tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from msomi.core.config import MsomiConfig, load_yaml_config


def test_default_config_loads():
    cfg = MsomiConfig()
    assert cfg.app.name == "Msomi"
    assert 0 < cfg.risk.per_trade_pct <= 1
    assert cfg.signals.min_confidence_score >= 0


def test_yaml_config_loads_from_file():
    path = Path(__file__).parents[1] / "config" / "settings.yaml"
    if not path.exists():
        pytest.skip("config/settings.yaml not found")
    cfg = load_yaml_config(path)
    assert isinstance(cfg, MsomiConfig)
    assert cfg.watchlist.forex
    assert cfg.watchlist.crypto


def test_risk_pct_validation():
    with pytest.raises(Exception):
        from msomi.core.config import RiskConfig
        RiskConfig(per_trade_pct=1.5)


def test_watchlist_all_symbols():
    cfg = MsomiConfig()
    all_syms = cfg.watchlist.all_symbols
    assert len(all_syms) == len(cfg.watchlist.forex) + len(cfg.watchlist.crypto)
