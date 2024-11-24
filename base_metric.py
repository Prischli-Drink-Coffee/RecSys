import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

from pandas import DataFrame as PandasDataFrame


MetricsDataFrameLike = Union[PandasDataFrame, Dict]
MetricsMeanReturnType = Mapping[str, float]
MetricsPerUserReturnType = Mapping[str, Mapping[Any, float]]
MetricsReturnType = Union[MetricsMeanReturnType, MetricsPerUserReturnType]


class MetricDuplicatesWarning(Warning):
    """Recommendations contain duplicates"""


class Metric(ABC):
    """Base metric class"""

    def __init__(
        self,
        topk: Union[List[int], int],
        query_column: str = "query_id",
        item_column: str = "item_id",
        rating_column: str = "rating"
    ) -> None:
        """
        :param topk: (list or int): Consider the highest k scores in the ranking.
        :param query_column: (str): The name of the user column.
        :param item_column: (str): The name of the item column.
        :param rating_column: (str): The name of the score column.
        :param mode: (CalculationDescriptor): class for calculating aggregation metrics.
            Default: ``Mean``.
        """
        if isinstance(topk, list):
            for item in topk:
                if not isinstance(item, int):
                    msg = f"{item} is not int"
                    raise ValueError(msg)
        elif isinstance(topk, int):
            topk = [topk]
        else:
            msg = "topk not list or int"
            raise ValueError(msg)
        self.topk = sorted(topk)
        self.query_column = query_column
        self.item_column = item_column
        self.rating_column = rating_column

    @property
    def __name__(self) -> str:
        mode_name = self._mode.__name__
        return str(type(self).__name__) + (f"-{mode_name}" if mode_name != "Mean" else "")

    def _check_dataframes_equal_types(
        self,
        recommendations: MetricsDataFrameLike,
        ground_truth: MetricsDataFrameLike,
    ) -> None:
        """
        Types of all data frames must be the same.
        """
        if not isinstance(recommendations, type(ground_truth)):
            msg = "All given data frames must have the same type"
            raise ValueError(msg)

    def _duplicate_warn(self):
        warnings.warn(
            "The recommendations contain duplicated users and items.The metrics may be higher than the actual ones.",
            MetricDuplicatesWarning,
        )

    def _check_duplicates_dict(self, recommendations: Dict) -> None:
        for items in recommendations.values():
            items_set = set(items)
            if len(items) != len(items_set):
                self._duplicate_warn()
                return

    def __call__(
        self,
        recommendations: MetricsDataFrameLike,
        ground_truth: MetricsDataFrameLike,
    ) -> MetricsReturnType:
        """
        Compute metric.

        :param recommendations: (PySpark DataFrame or Polars DataFrame or Pandas DataFrame or dict): model predictions.
            If DataFrame then it must contains user, item and score columns.
            If dict then key represents user_ids, value represents list of tuple(item_id, score).
        :param ground_truth: (PySpark DataFrame or Polars DataFrame or Pandas DataFrame or dict):
            test data.
            If DataFrame then it must contains user and item columns.
            If dict then key represents user_ids, value represents list of item_ids.

        :return: metric values
        """
        self._check_dataframes_equal_types(recommendations, ground_truth)
        is_pandas = isinstance(recommendations, PandasDataFrame)
        recommendations = (
            self._convert_pandas_to_dict_with_score(recommendations)
            if is_pandas
            else self._convert_dict_to_dict_with_score(recommendations)
        )
        self._check_duplicates_dict(recommendations)
        ground_truth = self._convert_pandas_to_dict_without_score(ground_truth) if is_pandas else ground_truth
        assert isinstance(ground_truth, dict)
        return self._dict_call(
            list(ground_truth),
            pred_item_id=recommendations,
            ground_truth=ground_truth,
        )

    def _convert_pandas_to_dict_with_score(self, data: PandasDataFrame) -> Dict:
        return (
            data.sort_values(by=self.rating_column, ascending=False)
            .groupby(self.query_column)[self.item_column]
            .apply(list)
            .to_dict()
        )

    def _convert_dict_to_dict_with_score(self, data: Dict) -> Dict:
        converted_data = {}
        for user, items in data.items():
            is_sorted = True
            for i in range(1, len(items)):
                is_sorted &= items[i - 1][1] >= items[i][1]
                if not is_sorted:
                    break
            if not is_sorted:
                items = sorted(items, key=lambda x: x[1], reverse=True)
            converted_data[user] = [item for item, _ in items]
        return converted_data

    def _convert_pandas_to_dict_without_score(self, data: PandasDataFrame) -> Dict:
        return data.groupby(self.query_column)[self.item_column].apply(list).to_dict()

    def _dict_call(self, users: List, **kwargs: Dict) -> MetricsReturnType:
        """
        Calculating metrics in dict format.
        kwargs can contain different dicts (for example, ground_truth or train), it depends on the metric.
        """

        keys_list = sorted(kwargs.keys())
        distribution_per_user = {}
        for user in users:
            args = [kwargs[key].get(user, None) for key in keys_list]
            distribution_per_user[user] = self._get_metric_value_by_user(self.topk, *args)
        if self._mode.__name__ == "PerUser":
            return self._aggregate_results_per_user(distribution_per_user)
        distribution = np.stack(list(distribution_per_user.values()))
        assert distribution.shape[1] == len(self.topk)
        metrics = [self._mode.cpu(distribution[:, k]) for k in range(distribution.shape[1])]
        return self._aggregate_results(metrics)

    def _aggregate_results_per_user(self, distribution_per_user: Dict[Any, List[float]]) -> MetricsPerUserReturnType:
        res: MetricsPerUserReturnType = {}
        for index, val in enumerate(self.topk):
            metric_name = f"{self.__name__}@{val}"
            res[metric_name] = {}
            for user, metrics in distribution_per_user.items():
                res[metric_name][user] = metrics[index]
        return res

    def _aggregate_results(self, metrics: list) -> MetricsMeanReturnType:
        res = {}
        for index, val in enumerate(self.topk):
            metric_name = f"{self.__name__}@{val}"
            res[metric_name] = metrics[index]
        return res

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(ks: List[int], *args: List) -> List[float]:  # pragma: no cover
        """
        Metric calculation for one user.

        :param k: depth cut-off
        :param ground_truth: test data
        :param pred: recommendations
        :return: metric value for current user
        """
        raise NotImplementedError()
