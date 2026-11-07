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


def test_run_json_is_strict_json_without_eve(capsys):
    """NaN metrics (no Eve, no Bell test) must become null, not the bare token NaN."""
    code = main(["run", "--protocol", "bb84", "-n", "100", "--seed", "1", "--json"])
    assert code == 0
    text = capsys.readouterr().out
    assert "NaN" not in text
    payload = json.loads(text, parse_constant=pytest.fail)  # reject NaN/Infinity tokens
    assert payload["eve_agreement"] is None
    assert payload["chsh"] is None
    assert payload["di_key_rate"] is None


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "-p", "bb84", "--noise", "1.5"],
        ["run", "-p", "bb84", "--noise", "-0.1"],
        ["run", "-p", "bb84", "--f-ec", "0.5"],
        ["run", "-p", "bb84", "-n", "-5"],
        ["run", "-p", "bb84", "--show", "-1"],
        ["sweep", "-p", "bb84", "--seeds", "0", "--out", "x.csv"],
        ["sweep", "-p", "bb84", "--f-ec", "0.9", "--out", "x.csv"],
    ],
)
def test_bad_arguments_are_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_help_names_the_installed_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage: qkd-simulation" in capsys.readouterr().out


def test_run_text_and_csv(tmp_path, capsys):
    out = tmp_path / "t.csv"
    code = main(
        [
            "run",
            "-p",
            "sarg04",
            "-n",
            "200",
            "--seed",
            "3",
            "--csv",
            str(out),
            "--show",
            "4",
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "SARG04" in text and "QBER" in text and "secret key rate" in text
    df = pd.read_csv(out)
    assert "sifted" in df.columns and len(df) == 200


def test_unknown_protocol_exits_with_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["run", "--protocol", "bbm92", "-n", "10"])
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
    assert len(table) == 12  # 4 protocols x 3 noise levels x 1 seed
    assert set(table["protocol"]) == {"b92", "bb84", "e91", "sarg04"}
    assert (table["theory_qber"].notna()).all()
    assert table["theory_chsh"].notna().sum() == 3
    e91_rows = table[table["protocol"] == "e91"]
    assert e91_rows["chsh"].notna().all()
    # ~90 Bell-test rounds give S a std of ~0.3; check the noiseless point to within 4 std.
    assert abs(e91_rows.loc[e91_rows["noise"] == 0, "chsh"].item() - 2 * 2**0.5) < 1.2
    assert "wrote 12 rows" in capsys.readouterr().out


def test_noise_spec_parsing():
    assert parse_noise_spec("0:0.1:0.05") == [0.0, 0.05, 0.1]
    assert parse_noise_spec("0.2,0.4") == [0.2, 0.4]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_noise_spec("0:2:0.5")


def test_noise_spec_never_overshoots_stop():
    assert parse_noise_spec("0:0.1:0.06") == [0.0, 0.06]
    assert parse_noise_spec("0.9:1:0.06") == [0.9, 0.96]
    assert parse_noise_spec("0:0.3:0.02")[-1] == 0.3  # a dividing step still reaches stop
    assert parse_noise_spec("0.1:0.1:0.05") == [0.1]
