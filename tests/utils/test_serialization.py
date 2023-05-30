from tempor.utils.serialization import load, load_from_file, save, save_to_file


def test_save_load() -> None:
    obj = {"a": 1, "b": "dssf"}

    objbytes = save(obj)

    reloaded = load(objbytes)

    assert isinstance(reloaded, dict)
    assert reloaded["a"] == 1
    assert reloaded["b"] == "dssf"


def test_save_load_file(tmpdir) -> None:
    path = tmpdir.mkdir("serialization").join("obj.p")
    obj = {"a": 1, "b": "dssf"}
    save_to_file(path, obj)
    reloaded = load_from_file(path)
    assert isinstance(reloaded, dict)
    assert reloaded["a"] == 1
    assert reloaded["b"] == "dssf"

    # Test the mkdir case also.
    path = str(tmpdir / "nonexistent" / "obj.p")
    obj = {"a": 1, "b": "dssf"}
    save_to_file(path, obj)
    reloaded = load_from_file(path)
    assert isinstance(reloaded, dict)
    assert reloaded["a"] == 1
    assert reloaded["b"] == "dssf"
