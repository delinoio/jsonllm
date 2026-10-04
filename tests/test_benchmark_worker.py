import importlib.util
import os
from pathlib import Path


def test_server_uses_its_own_build_tools_even_with_symlinked_environment(tmp_path):
    path = Path(__file__).parents[1] / "scripts/design_benchmark_worker.py"
    spec = importlib.util.spec_from_file_location("worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "isolated" / "bin"
    target.mkdir(parents=True)
    (target / "python").symlink_to("/usr/bin/python3")
    link = tmp_path / ".venv-vllm"
    link.symlink_to(target.parent)
    env = module.interpreter_environment(link / "bin" / "python")
    assert env["PATH"].split(os.pathsep)[0] == str(target)
    assert env["PATH"].endswith(os.environ["PATH"])
