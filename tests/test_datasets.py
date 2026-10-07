import pytest

from viz_agent import STORE, DatasetError, DatasetStore
from conftest import DATA


def test_list_and_resolve():
    assert STORE.list_files() == ["online-sales.csv", "students.csv"]
    assert STORE.resolve("") == "students.csv"              # default
    assert STORE.resolve("online-sales") == "online-sales.csv"
    assert STORE.resolve("STUDENTS.csv".lower()) == "students.csv"


@pytest.mark.parametrize("name", ["../secrets.csv", "..\\..\\x.csv", "/etc/passwd", "nope.csv"])
def test_resolve_never_leaves_data_folder(name):
    with pytest.raises(DatasetError, match="not found"):
        STORE.resolve(name)


def test_no_default_and_several_files_asks_which():
    with pytest.raises(DatasetError, match="Which dataset"):
        DatasetStore(DATA).resolve("")


def test_single_file_folder_is_the_default(tmp_path):
    (tmp_path / "only.csv").write_text("a,b\n1,2\n")
    assert DatasetStore(tmp_path).resolve(None) == "only.csv"


def test_cache_loads_once():
    assert STORE.load("students.csv") is STORE.load("students")


def test_unreadable_csv(tmp_path):
    store = DatasetStore(tmp_path)
    (tmp_path / "bad.csv").write_text('a,b\n"1,2\n')  # unterminated quote
    with pytest.raises(DatasetError, match="Could not read"):
        store.load("bad.csv")


def test_column_validation_is_case_insensitive():
    df = STORE.load()
    assert STORE.column(df, "g3", numeric=True) == "G3"
    with pytest.raises(DatasetError, match="not found"):
        STORE.column(df, "grade")
    with pytest.raises(DatasetError, match="not numeric"):
        STORE.column(df, "sex", numeric=True)
    with pytest.raises(DatasetError, match="required"):
        STORE.column(df, "")


def test_summary():
    s = STORE.summary("students.csv")
    assert s["rows"] == 400 and len(s["columns"]) == 17
    g3 = next(c for c in s["columns"] if c["name"] == "G3")
    assert g3["dtype"] == "int64" and isinstance(g3["example"], int)


def test_sample_data_tells_the_lab_story():
    df = STORE.load()
    by_medu = df.groupby("Medu")["G3"].mean()
    assert by_medu.is_monotonic_increasing                               # Exercise 1: parents' education matters
    assert df.groupby("internet")["G3"].mean()["yes"] > df.groupby("internet")["G3"].mean()["no"]  # Exercise 2
    assert -0.2 < df["absences"].corr(df["G3"]) < 0                       # Exercise 3: weak negative
