from typing import Optional

from pandas import DataFrame as PandasDataFrame



def groupby_sequences(events: PandasDataFrame, groupby_col: str, sort_col: Optional[str] = None) -> PandasDataFrame:
    """
    :param events: dataframe with interactions
    :param groupby_col: divide column to group by
    :param sort_col: column to sort by

    :returns: dataframe with sequences for each value in groupby_col
    """
    if isinstance(events, PandasDataFrame):
        event_cols_without_groupby = events.columns.values.tolist()
        event_cols_without_groupby.remove(groupby_col)

        if sort_col:
            event_cols_without_groupby.remove(sort_col)
            event_cols_without_groupby.insert(0, sort_col)
            events = events.sort_values(event_cols_without_groupby)

        grouped_sequences = (
            events.groupby(groupby_col).agg({col: list for col in event_cols_without_groupby}).reset_index()
        )
    return grouped_sequences


def ensure_pandas(
    df: PandasDataFrame,
    allow_collect_to_master: bool = False,
) -> PandasDataFrame:
    """
    :param df: dataframe
    :param allow_collect_to_master: Flag allowing spark to make a collection to the master node,
        default: ``False``.

    :returns: Pandas DataFrame object
    """
    if isinstance(df, PandasDataFrame):
        return df
