import importlib

def test_import_and_app():
    mod = importlib.import_module('nlp_automl_optuna' + ".cli")
    assert hasattr(mod, "app")
