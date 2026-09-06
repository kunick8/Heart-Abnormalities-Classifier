import pandas as pd

from src.data.data_cleaning import join_data


def test_join_data_concatenates_rows():
    data = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "feature_1": [10, 20, 30, 40],
        "feature_2": [1, 2, 3, 4],
    })

    data2 = pd.DataFrame({
        "target": [0, 2, 0, 2],
        "feature_1": [20, 30, 40, 50],
        "feature_2": [2, 3, 4, 5],
    })

    joined = join_data(data, data2)

    assert joined.shape == (8, 3)
    assert list(joined.columns) == ["target", "feature_1", "feature_2"]
    assert len(joined) == len(data) + len(data2)


def test_join_data_preserves_all_rows_from_both_frames():
    data = pd.DataFrame({"target": [0], "feature_1": [1], "feature_2": [2]})
    data2 = pd.DataFrame({"target": [1], "feature_1": [3], "feature_2": [4]})

    joined = join_data(data, data2)

    assert 0 in joined["target"].values
    assert 1 in joined["target"].values
    assert set(joined["feature_1"]) == {1, 3}