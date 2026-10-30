from overtone_core import config


def test_a_source_checkout_is_rooted_at_the_repo(tmp_path):
    (tmp_path / "packages" / "core").mkdir(parents=True)
    (tmp_path / "packages" / "core" / "pyproject.toml").touch()
    module = tmp_path / "packages" / "core" / "src" / "overtone_core" / "config.py"

    assert config.find_root(module, prefix=tmp_path / "some-venv") == (str(tmp_path.resolve()), False)


def test_an_install_is_rooted_at_the_folder_holding_its_venv(tmp_path):
    venv = tmp_path / "Overtone" / ".venv"
    module = venv / "Lib" / "site-packages" / "overtone_core" / "config.py"

    assert config.find_root(module, prefix=venv) == (str((tmp_path / "Overtone").resolve()), True)


def test_the_tests_run_from_a_checkout():
    assert not config.INSTALLED
    assert config.WEIGHTS_DIR is None  # a checkout keeps the default torch/Hugging Face caches
