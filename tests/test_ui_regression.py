from jsonllm.genui_suite import assemble, build_suite
from jsonllm.ui import compile_ui, run_ui
from jsonllm.ui_examples import workflow_spec


def test_registered_rules_reproduce_all_sixty_independent_legacy_cases_without_model():
    count = 0
    for trajectory in build_suite()["trajectories"]:
        compiled = compile_ui(workflow_spec(trajectory["flow"], trajectory["registry"]))
        state = trajectory["initial_state"]
        for step in trajectory["steps"]:
            result = run_ui(compiled, "", state, step["event"], step["facts"])
            assert result["output"] == assemble(step["expected"]), (trajectory["id"], result)
            assert not result["diagnostics"]["model_fields"]
            state = result["output"]["state"]
            count += 1
    assert count == 60
