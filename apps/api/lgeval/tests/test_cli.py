from __future__ import annotations

import sys
import types

import pytest

from lgeval import cli
from lgeval.cli import SUITES, build_parser, main, plan_rows


def test_plan_pilot_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--suite", "pilot", "--offline"]) == 0
    out = capsys.readouterr().out
    for word in ("full", "no_debate", "no_critic", "single_shot", "single_agent", "distrust", "identical", "total"):
        assert word in out


def test_plan_rows_math() -> None:
    rows = plan_rows(SUITES["pilot"], run_usd=1.0)
    by = {r.detail.split(":")[0]: r for r in rows}
    assert by["full"].runs == 10 and by["single_agent"].runs == pytest.approx(3.0)
    assert by["no_debate"].runs == pytest.approx(6.0)
    assert by["steer"].runs == pytest.approx(5 * 1.35) and by["steer"].briefs == 10
    score = rows[-1]
    assert score.experiment == "score" and score.usd == pytest.approx(100 * 0.03)
    full = plan_rows(SUITES["full"], run_usd=1.0)
    assert full[0].runs == 90 and sum(r.experiment == "recompute" for r in full) == 4


def test_plugins_register_and_missing_are_noted(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    mod = types.ModuleType("lgeval_fake_plugin")

    async def handler(args) -> int:
        print(f"hello {args.name}")
        return 3

    def register(sub) -> None:
        p = sub.add_parser("hello", help="fake")
        p.add_argument("--name", default="x")
        p.set_defaults(func=handler)

    mod.register = register
    monkeypatch.setitem(sys.modules, "lgeval_fake_plugin", mod)
    monkeypatch.setattr(cli, "PLUGINS", ("lgeval_fake_plugin", "lgeval_does_not_exist"))
    parser = build_parser(cli.PLUGINS)
    assert "lgeval_does_not_exist" in (parser.epilog or "")
    assert main(["hello", "--name", "bob"]) == 3
    assert "hello bob" in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "plan" in capsys.readouterr().out
