import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
from pandas import DataFrame as PandasDataFrame


SplitterReturnType = Tuple[PandasDataFrame, PandasDataFrame]


class Splitter(ABC):
    """Base class"""

    _init_arg_names = [
        "drop_cold_users",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    def __init__(
        self,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        query_column: str = "query_id",
        item_column: Optional[str] = "item_id",
        timestamp_column: Optional[str] = "timestamp",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param drop_cold_items: flag to remove items that are not in train data
        :param drop_cold_users: flag to remove users that are not in train data
        :param query_column: query id column name
        :param item_column: item id column name
        :param timestamp_column: timestamp column name
        :param session_id_column: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        self.drop_cold_users = drop_cold_users
        self.drop_cold_items = drop_cold_items
        self.query_column = query_column
        self.item_column = item_column
        self.timestamp_column = timestamp_column
        self.session_id_column = session_id_column
        self.session_id_processing_strategy = session_id_processing_strategy

    @property
    def _init_args(self):
        return {name: getattr(self, name) for name in self._init_arg_names}

    def save(self, path: str) -> None:
        """
        Method for saving splitter in `.replay` directory.
        """
        base_path = Path(path).with_suffix(".replay").resolve()
        base_path.mkdir(parents=True, exist_ok=True)
        splitter_dict = {"init_args": self._init_args, "_class_name": str(self)}
        with open(base_path / "init_args.json", "w+") as file:
            json.dump(splitter_dict, file)

    @classmethod
    def load(cls, path: str, **kwargs) -> "Splitter":
        """
        Method for loading splitter from `.replay` directory.
        """
        base_path = Path(path).with_suffix(".replay").resolve()
        with open(base_path / "init_args.json", "r") as file:
            splitter_dict = json.loads(file.read())
        splitter = cls(**splitter_dict["init_args"])
        return splitter

    def __str__(self):
        return type(self).__name__

    def _drop_cold_items_and_users(
            self,
            train: PandasDataFrame,
            test: PandasDataFrame,
    ) -> Tuple[PandasDataFrame, Optional[PandasDataFrame], Optional[PandasDataFrame]]:
        if isinstance(train, type(test)) is False:
            msg = "Train and test dataframes must have consistent types"
            raise TypeError(msg)
        if isinstance(test, PandasDataFrame):
            return self._drop_cold_items_and_users_from_pandas(train, test)

    def _drop_cold_items_and_users_from_pandas(
            self,
            train: PandasDataFrame,
            test: PandasDataFrame,
    ) -> Tuple[PandasDataFrame, Optional[PandasDataFrame], Optional[PandasDataFrame]]:
        cold_items_df = None
        cold_users_df = None
        if self.drop_cold_items:
            cold_items_mask = ~test[self.item_column].isin(train[self.item_column])
            cold_items_df = test[cold_items_mask]
            test = test[~cold_items_mask]
        if self.drop_cold_users:
            cold_users_mask = ~test[self.query_column].isin(train[self.query_column])
            cold_users_df = test[cold_users_mask]
            test = test[~cold_users_mask]
        return test, cold_users_df, cold_items_df

    @abstractmethod
    def _core_split(self, interactions: PandasDataFrame) -> SplitterReturnType:
        """
        This method implements split strategy

        :param interactions: input DataFrame `[timestamp, user_id, item_id, relevance]`
        :returns: `train` and `test DataFrames
        """

    def split(self, interactions: PandasDataFrame) -> Tuple[SplitterReturnType, Optional[PandasDataFrame], Optional[PandasDataFrame]]:
        """
        Splits input DataFrame into train and test, and optionally returns DataFrames with cold users/items.

        :param interactions: input DataFrame `[timestamp, user_id, item_id, relevance]`.
        :returns: Tuple of (train, test DataFrames), DataFrame of cold users, DataFrame of cold items.
        """
        train, test = self._core_split(interactions)
        test, cold_users_df, cold_items_df = self._drop_cold_items_and_users(train, test)
        return (train, test), cold_users_df, cold_items_df

    def _recalculate_with_session_id_column(self, data: PandasDataFrame) -> PandasDataFrame:
        return self._recalculate_with_session_id_column_pandas(data)

    def _recalculate_with_session_id_column_pandas(self, data: PandasDataFrame) -> PandasDataFrame:
        agg_function_name = "first" if self.session_id_processing_strategy == "train" else "last"
        res = data.copy()
        res["is_test"] = res.groupby([self.query_column, self.session_id_column])["is_test"].transform(
            agg_function_name
        )
        return res
