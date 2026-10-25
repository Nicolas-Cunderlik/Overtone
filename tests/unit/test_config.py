from overtone import config


def test_migrates_the_legacy_data_dir_once(tmp_path):
    legacy, new = tmp_path / "Spotistats", tmp_path / "Overtone"
    (legacy / "cache").mkdir(parents=True)
    (legacy / "spotify_token.json").write_text("{}")

    config.migrate_legacy_data_dir(str(new), str(legacy))

    assert not legacy.exists()
    assert (new / "spotify_token.json").read_text() == "{}"
    assert (new / "cache").is_dir()


def test_leaves_both_dirs_alone_when_the_new_one_exists(tmp_path):
    legacy, new = tmp_path / "Spotistats", tmp_path / "Overtone"
    legacy.mkdir()
    new.mkdir()

    config.migrate_legacy_data_dir(str(new), str(legacy))

    assert legacy.is_dir() and new.is_dir()


def test_does_nothing_without_a_legacy_dir(tmp_path):
    config.migrate_legacy_data_dir(str(tmp_path / "Overtone"), str(tmp_path / "Spotistats"))
    assert not (tmp_path / "Overtone").exists()


def test_tests_never_use_the_real_data_dir():
    assert "overtone-test-" in config.APP_DATA_DIR  # set by tests/conftest.py
