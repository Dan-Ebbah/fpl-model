"""Snapshot store: persisting a pre-fetched pull without touching the network."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from fpl_model import api


def test_snapshot_persists_prefetched_data(tmp_path):
    bootstrap = {
        "elements": [{"code": 1, "web_name": "A", "now_cost": 50}],
        "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}],
    }
    fixtures = [{"event": 1, "team_h": 1, "team_a": 2}]

    dest = api.snapshot(out_dir=tmp_path, bootstrap=bootstrap, fixtures=fixtures)

    assert dest.name == dt.date.today().isoformat()
    for f in ["bootstrap.json", "players.parquet", "teams.parquet", "fixtures.parquet"]:
        assert (dest / f).exists()

    assert pd.read_parquet(dest / "players.parquet").iloc[0]["code"] == 1
    assert pd.read_parquet(dest / "fixtures.parquet").iloc[0]["event"] == 1

    # same-day re-snapshot overwrites in place rather than duplicating
    assert api.snapshot(out_dir=tmp_path, bootstrap=bootstrap, fixtures=fixtures) == dest


def test_failed_snapshot_leaves_no_folder(tmp_path, monkeypatch):
    def boom():
        raise ConnectionError("TLS EOF")

    monkeypatch.setattr(api, "fetch_bootstrap", boom)
    with pytest.raises(ConnectionError):
        api.snapshot(out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
