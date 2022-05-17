import pytest
import numpy as np
import pandas as pd
from ..algorithm import OutlierDetectionModel

np.random.seed(42)


def test_compute_rolling_window_stats():
    # arrange
    lookback = 5
    min_obs = 1
    data = pd.Series(np.random.rand(20))
    exp_mu = data.rolling(lookback, min_obs).mean()
    exp_sigma = data.rolling(lookback, min_obs).std()
    exp_target = data

    # act
    res_df = OutlierDetectionModel._compute_rolling_window_stats(
        data, lookback, min_obs
    )

    # assert
    assert res_df.target.equals(exp_target)
    assert res_df.mu.equals(exp_mu)
    assert res_df.sigma.equals(exp_sigma)


@pytest.mark.parametrize(
    "inc_curr, expected",
    [
        (True, pd.Series([310, 3260, 803, 993, -476])),
        (False, pd.Series([0, 1000, 1173, 624, -234])),
    ],
)
def test_add_z_scores(inc_curr, expected):
    np.random.seed(42)
    target = np.random.rand(5)

    np.random.seed(43)
    mu = np.random.rand(5)

    np.random.seed(44)
    sigma = np.random.rand(5)

    df = pd.DataFrame(data={"target": target, "mu": mu, "sigma": sigma})
    OutlierDetectionModel._add_z_scores(df, inc_curr)
    assert df.z_score.apply(lambda x: int(x * 1000)).equals(expected)


@pytest.mark.parametrize(
    "z_score_vector, threshold, expected",
    [
        (pd.Series([0.3, 0.4, 3.5]), 3, pd.Series([False, False, True])),
        (pd.Series([0.0, 0.0, 0.0]), 4, pd.Series([False, False, False])),
        (pd.Series([0.1, 0.0, 0.1]), 0, pd.Series([True, False, True])),
        (pd.Series([-3.4, 4.5, -2.4, 2.3]), 3, pd.Series([True, True, False, False])),
    ],
)
def test_determine_outlier(z_score_vector, threshold, expected):
    assert OutlierDetectionModel._determine_outlier(
        z_scores=z_score_vector, z_score_threshold=threshold
    ).equals(expected)


# error cases
@pytest.mark.parametrize(
    "data,lookback,min_obs,expected",
    [
        (pd.Series(np.random.rand(0)), 5, 1, AssertionError),  # no data
        (pd.Series(np.random.rand(2)), 5, 1, AssertionError),  # no data
        (pd.Series(["I'm", "a", "string"]), 5, 1, TypeError),
    ],
)
def test_compute_rolling_window_stats_error_cases(data, lookback, min_obs, expected):
    # act
    with pytest.raises(expected):
        OutlierDetectionModel._compute_rolling_window_stats(data, lookback, min_obs)
