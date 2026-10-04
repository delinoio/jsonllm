def test_unknown_config_key_fails(tmp_path):
    import pytest

    from jsonllm.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text("unknown: true\n")
    with pytest.raises(ValueError, match="Invalid training config"):
        load_config(path)


def test_explicit_env_file_parses_without_shell_execution(tmp_path):
    from jsonllm.config import read_env_file

    path = tmp_path / ".env"
    path.write_text(
        "# example\nexport OPENROUTER_API_KEY='secret#value'\n"
        "OPENROUTER_MODEL=teacher/model # comment\nIGNORED='$(touch bad)'\n"
    )
    values = read_env_file(path)
    assert values["OPENROUTER_API_KEY"] == "secret#value"
    assert values["OPENROUTER_MODEL"] == "teacher/model"
    assert values["IGNORED"] == "$(touch bad)"
