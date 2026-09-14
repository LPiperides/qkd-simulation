import argparse
import json

import pandas as pd
import pytest

from qkd_sim.cli import main, parse_noise_spec


def test_run_json(capsys):
    code = main(["run", "--protocol", "bb84", "-n", "300", "--eve", "--seed", "1", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"qber", "retention", "key_rate", "i_ab", "i_ae", "theory_qber"} <= set(payload)
    assert payload["with_eve"] is True
    assert payload["theory_qber"] == 0.25


def test_run_text_and_csv(tmp_path, capsys):
    out = tmp_path / "t.csv"
    code = main(
        ["run", "-p", "sarg04", "-n", "200", "--seed", "3", "--csv", str(out), "--show", "4"]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "SARG04" in text and "QBER" in text and "secret key rate" in text
    df = pd.read_csv(out)
    assert "sifted" in df.columns and len(df) == 200


def test_unknown_protocol_exits_with_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["run", "--protocol", "e91", "-n", "10"])
    assert exc.value.code == 2


def test_sweep_writes_expected_rows(tmp_path, capsys):
    out = tmp_path / "s.csv"
    code = main(
        [
            "sweep",
            "--protocol",
            "all",
            "--noise",
            "0:0.1:0.05",
            "--seeds",
            "1",
            "-n",
            "200",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    table = pd.read_csv(out)
    assert len(table) == 9  # 3 protocols x 3 noise levels x 1 seed
    assert set(table["protocol"]) == {"b92", "bb84", "sarg04"}
    assert (table["theory_qber"].notna()).all()
    assert "wrote 9 rows" in capsys.readouterr().out


def test_noise_spec_parsing():
    assert parse_noise_spec("0:0.1:0.05") == [0.0, 0.05, 0.1]
    assert parse_noise_spec("0.2,0.4") == [0.2, 0.4]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_noise_spec("0:2:0.5")
