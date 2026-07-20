from pathlib import Path

from evipatch.runner import load_stage_config


def test_physionet_first_clean_tuple_order_matches_deserializer() -> None:
    config = load_stage_config()
    source = (
        Path(config["project"]["apn_root"])
        / "data/dependencies/tsdm/datasets/physionet2012.py"
    ).read_text(encoding="utf-8")
    assert "self.dataset[key] = time_series_df, metadata_df" in source
    assert "return time_series_df, metadata_df" in source
