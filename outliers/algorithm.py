from .interfaces import UnivariateModel
import pandas as pd
from pydantic import BaseModel


DEFAULT_LOOKBACK_WINDOW = 26
DEFAULT_Z_SCORE_THRESHOLD = 3
DEFAULT_MIN_OBS = 1
DEFAULT_Z_SCORE_INCL_CURRENT = False


class TrailingZScoreConfig(BaseModel):
    """
    Configuration data class to be provided
    to the class constructor for the OutlierDetectionModel
    im
    """

    lookback_window: int = DEFAULT_LOOKBACK_WINDOW
    z_score_threshold: int = DEFAULT_Z_SCORE_THRESHOLD
    min_obs: int = DEFAULT_MIN_OBS
    z_score_incl_current: bool = DEFAULT_Z_SCORE_INCL_CURRENT

    class Config:
        validate_assignment = True


class OutlierDetectionModel(UnivariateModel):
    """
    Implementation of the outlier detection model using
    trailing z scores.
    This algorithm accepts two strategies with regard to z scores. One taking the
    contribution to the mean and standard deviation of the observation
    being assessed into account and the other, non-standard approach of
    calculating the z score of the new observation relative to the mean and
    std deviation figures from the previous observation. In testing this
    showed a good ability to pick out subtle outliers, although this can come
    with additional noise. Strategy choice is determined by the z_score_incl_current
    configuration.
    """

    def __init__(self, config: TrailingZScoreConfig):
        self.config: TrailingZScoreConfig = config

    def fit_predict(self, data: "pd.Series[float]") -> "pd.Series[bool]":
        """
        Takes in the single discrete variable and for each observation
        predict whether the observation is an outlier (1) or not an outlier
        (0)
        """
        computed_df = self._compute_rolling_window_stats(
            data=data,
            lookback_window=self.config.lookback_window,
            min_obs=self.config.min_obs,
        )
        self._add_z_scores(
            df=computed_df, incl_current=self.config.z_score_incl_current
        )
        return self._determine_outlier(
            z_scores=computed_df.z_score,
            z_score_threshold=self.config.z_score_threshold,
        )

    @staticmethod
    def _compute_rolling_window_stats(
        data: "pd.Series[float]", lookback_window: int, min_obs: int
    ) -> pd.DataFrame:
        try:
            data = data.astype("float64")  # type: ignore
        except:
            raise TypeError(
                "Cannot convert series to float. Ensure vector contains "
                "valid numerical data"
            )
        assert len(data) > 2, "Data size must be > 2 in order to assess outliers"

        rolling = data.rolling(lookback_window, min_obs)  # type: ignore
        mu = rolling.mean()
        sigma = rolling.std()
        return pd.DataFrame(data={"target": data, "mu": mu, "sigma": sigma})

    @staticmethod
    def _add_z_scores(df: pd.DataFrame, incl_current: bool) -> None:
        # make zeroes 1 to avoid div by 0
        df.sigma[df.sigma == 0] = 1  # type: ignore

        if incl_current:
            df["z_score"] = (df.target - df.mu) / df.sigma  # type: ignore
        else:
            df["z_score"] = [0] + [
                (df.target[i] - df.mu[i - 1]) / (df.sigma[i - 1])
                for i in range(1, len(df))
            ]

    @staticmethod
    def _determine_outlier(z_scores: pd.Series, z_score_threshold: int) -> pd.Series:
        return z_scores.abs() > z_score_threshold  # type: ignore
