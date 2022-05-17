from abc import ABCMeta, abstractmethod
import pandas as pd
from pydantic import BaseModel


class UnivariateModel(metaclass=ABCMeta):
    """
    Interface for a univariate model. Model should be
    instantiated with a config file
    """

    def __init__(self, config: BaseModel):
        self.config = config

    @abstractmethod
    def fit_predict(self, data: pd.Series) -> pd.Series:
        """
        This method should return a series with the prediction
        given a single dependent variable vector
        """
        ...
