from tinyagent import cli


def test_cli_single_query(server, capsys, tmp_path):
    (tmp_path / "hello.txt").write_text("hi")
    cli.main(["--plain", "--yes", "-w", str(tmp_path), "-q", "Read hello.txt"])
    out = capsys.readouterr().out
    assert "ACTION" in out and "OBSERVATION" in out and "ANSWER" in out


def test_cli_interactive_commands(server, capsys, monkeypatch, tmp_path):
    inputs = iter(["/help", "/tools", "/skills", "/model", "/memory", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli.main(["--plain", "-w", str(tmp_path)])
    out = capsys.readouterr().out
    assert "/clear" in out and "execute_python" in out and "Goodbye" in out
