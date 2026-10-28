from overtone import config


def test_tests_never_use_the_real_data_dir():
    assert "overtone-test-" in config.APP_DATA_DIR  # set by tests/conftest.py
