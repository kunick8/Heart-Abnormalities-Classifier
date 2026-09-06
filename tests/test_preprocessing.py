import numpy as np
import pytest
from sklearn.feature_selection import RFECV, SelectFromModel

from src.data.data_preprocessing import *


def test_data_splitter(small_dataset):
    X, y = data_splitter(small_dataset)
    assert X.shape == (4, 2)
    assert y.shape == (4,)
    assert np.array_equal(y, np.array([0, 1, 0, 1]))


def test_train_test_split(medium_dataset):
    X, y = data_splitter(medium_dataset)
    X_train, X_test, y_train, y_test = training_test_split(X, y, test_size=0.3)

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3


def test_train_test_split_is_stratified(medium_dataset):
    X, y = data_splitter(medium_dataset)
    _, _, y_train, y_test = training_test_split(X, y, test_size=0.4)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)


def test_feature_scaling(medium_dataset):
    X, y = data_splitter(medium_dataset)
    X_train, X_test, _, _ = training_test_split(X, y)
    X_train, X_test, scaler = feature_scaling(X_train, X_test)

    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_train.std(axis=0), 1, atol=1e-7)
    assert hasattr(scaler, "mean_")


def test_feature_scaling_uses_train_statistics_on_test_set(medium_dataset):
    X, y = data_splitter(medium_dataset)
    X_train, X_test, _, _ = training_test_split(X, y)
    _, X_test_scaled, scaler = feature_scaling(X_train, X_test)

    manual = (X_test - scaler.mean_) / scaler.scale_
    assert np.allclose(X_test_scaled, manual, atol=1e-8)


def test_smote_resampling_balances_classes(imbalanced_dataset):
    X, y = data_splitter(imbalanced_dataset)
    X_train, _, y_train, _ = training_test_split(X, y)

    X_res, y_res, _ = smote_resampling(X_train, y_train)

    counts = np.bincount(y_res)
    assert counts[0] == counts[1]
    assert X_res.shape[1] == X_train.shape[1]


def test_smote_resampling_respects_custom_ratio(imbalanced_dataset):
    X, y = data_splitter(imbalanced_dataset)
    X_train, _, y_train, _ = training_test_split(X, y)

    _, y_res, _ = smote_resampling(X_train, y_train, smote_ratio=0.6)

    counts = np.bincount(y_res)
    assert counts[1] == pytest.approx(0.6 * counts[0], abs=1)


def test_feature_selection(large_dataset):
    X, y = data_splitter(large_dataset)
    X_train, X_test, y_train, _ = training_test_split(X, y)
    X_train, X_test, _ = feature_scaling(X_train, X_test)

    X_train_sel, X_test_sel, selector = feature_selection(X_train, X_test, y_train)

    assert X_train_sel.shape[0] == X_train.shape[0]
    assert X_test_sel.shape[0] == X_test.shape[0]
    assert X_train_sel.shape[1] == X_test_sel.shape[1]
    assert X_train_sel.shape[1] <= X_train.shape[1]

    assert isinstance(selector, RFECV)
    assert hasattr(selector, "support_")


def test_ann_feature_selection_default(large_dataset):
    X, y = data_splitter(large_dataset)
    X_train, X_test, y_train, _ = training_test_split(X, y)
    X_train, X_test, _ = feature_scaling(X_train, X_test)

    X_train_sel, X_test_sel, selector = ann_feature_selection(X_train, X_test, y_train)

    assert X_train_sel.shape[1] == X_test_sel.shape[1]
    assert 1 <= X_train_sel.shape[1] <= X_train.shape[1]
    assert isinstance(selector, SelectFromModel)


def test_ann_feature_selection_respects_max_features(large_dataset):
    X, y = data_splitter(large_dataset)
    X_train, X_test, y_train, _ = training_test_split(X, y)
    X_train, X_test, _ = feature_scaling(X_train, X_test)

    X_train_sel, X_test_sel, _ = ann_feature_selection(
        X_train, X_test, y_train, n_features=1
    )

    assert X_train_sel.shape[1] == 1
    assert X_test_sel.shape[1] == 1


def test_get_optimization_data_matches_plain_split(medium_dataset):
    X, y = data_splitter(medium_dataset)
    X_train, X_test, y_train, y_test = get_optimization_data(medium_dataset)
    X_train_ref, X_test_ref, y_train_ref, y_test_ref = training_test_split(X, y)

    assert np.array_equal(X_train, X_train_ref)
    assert np.array_equal(X_test, X_test_ref)
    assert np.array_equal(y_train, y_train_ref)
    assert np.array_equal(y_test, y_test_ref)


def test_get_optimization_data_is_unscaled(medium_dataset):
    X_train, _, _, _ = get_optimization_data(medium_dataset)
    assert not np.allclose(X_train.std(axis=0), 1, atol=1e-6)