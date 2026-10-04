"""Catch missing dependencies when extracting the package from the experiment repo."""

import importlib
import pkgutil

import jsonllm


def test_all_library_modules_import_without_loading_models():
    for module in pkgutil.walk_packages(jsonllm.__path__, jsonllm.__name__ + "."):
        if module.name != "jsonllm.__main__":
            importlib.import_module(module.name)
