from typing import Any, List, Literal, Optional, Protocol, Tuple
from typing import Generic, List, Optional, Protocol, Tuple, TypeVar, cast
import abc
import lightning
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from replay.metrics.torch_metrics_builder import TorchMetricsBuilder, metrics_to_df
from _base import BasePostProcessor
from lightning_bert import Bert4Rec

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch_metrics_builder import TorchMetricsBuilder, metrics_to_df

from pandas import DataFrame as PandasDataFrame

CallbackMetricName = Literal[
    "recall",
    "precision",
    "ndcg",
    "map",
    "mrr"
]


class ValidationBatch(Protocol):
    """
    Validation callback batch
    """

    query_id: torch.LongTensor
    ground_truth: torch.LongTensor
    train: torch.LongTensor


class ValidationMetricsCallback(lightning.Callback):
    """
    Callback for validation and testing stages.

    If multiple validation/testing dataloaders are used,
    the suffix of the metric name will contain the serial number of the dataloader.
    """

    def __init__(
            self,
            metrics: Optional[List[CallbackMetricName]] = None,
            ks: Optional[List[int]] = None,
            postprocessors: Optional[List[BasePostProcessor]] = None,
            item_count: Optional[int] = None,
    ):
        """
        :param metrics: Sequence of metrics to calculate.
        :param ks: highest k scores in ranking. Default: will be `[1, 5, 10, 20]`.
        :param postprocessors: postprocessors to validation stage.
        :param item_count: the total number of items in the dataset, required only for Coverage calculations.
        """
        self._metrics = metrics
        self._ks = ks
        self._item_count = item_count
        self._metrics_builders: List[TorchMetricsBuilder] = []
        self._dataloaders_size: List[int] = []
        self._postprocessors: List[BasePostProcessor] = postprocessors or []

    def _get_dataloaders_size(self, dataloaders: Optional[Any]) -> List[int]:
        if isinstance(dataloaders, torch.utils.data.DataLoader):
            return [len(dataloaders)]
        return [len(dataloader) for dataloader in dataloaders]

    def on_validation_epoch_start(
            self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:
        self._dataloaders_size = self._get_dataloaders_size(trainer.val_dataloaders)
        self._metrics_builders = [
            TorchMetricsBuilder(self._metrics, self._ks, self._item_count) for _ in self._dataloaders_size
        ]
        for builder in self._metrics_builders:
            builder.reset()

    def on_test_epoch_start(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,  # noqa: ARG002
    ) -> None:  # pragma: no cover
        self._dataloaders_size = self._get_dataloaders_size(trainer.test_dataloaders)
        self._metrics_builders = [
            TorchMetricsBuilder(self._metrics, self._ks, self._item_count) for _ in self._dataloaders_size
        ]
        for builder in self._metrics_builders:
            builder.reset()

    def _compute_pipeline(
            self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        for postprocessor in self._postprocessors:
            query_ids, scores, ground_truth = postprocessor.on_validation(query_ids, scores, ground_truth)
        return query_ids, scores, ground_truth

    def on_validation_batch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            outputs: torch.Tensor,
            batch: ValidationBatch,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            outputs: torch.Tensor,
            batch: ValidationBatch,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:  # pragma: no cover
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _batch_end(
            self,
            trainer: lightning.Trainer,  # noqa: ARG002
            pl_module: lightning.LightningModule,
            outputs: torch.Tensor,
            batch: ValidationBatch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        _, seen_scores, seen_ground_truth = self._compute_pipeline(batch.query_id, outputs, batch.ground_truth)
        sampled_items = torch.topk(seen_scores, k=self._metrics_builders[dataloader_idx].max_k, dim=1).indices
        self._metrics_builders[dataloader_idx].add_prediction(sampled_items, seen_ground_truth, batch.train)

        if batch_idx + 1 == self._dataloaders_size[dataloader_idx]:
            pl_module.log_dict(
                self._metrics_builders[dataloader_idx].get_metrics(),
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=True,
            )

    def on_validation_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        self._epoch_end(trainer, pl_module)

    def on_test_epoch_end(
            self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:  # pragma: no cover
        self._epoch_end(trainer, pl_module)

    def _epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:  # noqa: ARG002
        @rank_zero_only
        def print_metrics() -> None:
            metrics = {}
            for name, value in trainer.logged_metrics.items():
                if "@" in name:
                    metrics[name] = value.item()

            if metrics:
                metrics_df = metrics_to_df(metrics)

                print(metrics_df)  # noqa: T201
                print()  # noqa: T201

        print_metrics()


class PredictionBatch(Protocol):
    """
    Prediction callback batch
    """

    query_id: torch.LongTensor


_T = TypeVar("_T")


class BasePredictionCallback(lightning.Callback, Generic[_T]):
    """
    Base callback for prediction stage
    """

    def __init__(
            self,
            top_k: int,
            query_column: str,
            item_column: str,
            rating_column: str = "rating",
            postprocessors: Optional[List[BasePostProcessor]] = None,
    ) -> None:
        """
        :param top_k: Takes the highest k scores in the ranking.
        :param query_column: query column name.
        :param item_column: item column name.
        :param rating_column: rating column name.
        :param postprocessors: postprocessors to apply.
        """
        super().__init__()
        self.query_column = query_column
        self.item_column = item_column
        self.rating_column = rating_column
        self._top_k = top_k
        self._postprocessors: List[BasePostProcessor] = postprocessors or []
        self._query_batches: List[torch.Tensor] = []
        self._item_batches: List[torch.Tensor] = []
        self._item_scores: List[torch.Tensor] = []

    def on_predict_epoch_start(
            self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:
        self._query_batches.clear()
        self._item_batches.clear()
        self._item_scores.clear()

    def on_predict_batch_end(
            self,
            trainer: lightning.Trainer,  # noqa: ARG002
            pl_module: lightning.LightningModule,  # noqa: ARG002
            outputs: torch.Tensor,
            batch: PredictionBatch,
            batch_idx: int,  # noqa: ARG002
            dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        query_ids, scores = self._compute_pipeline(batch.query_id, outputs)
        top_scores, top_item_ids = torch.topk(scores, k=self._top_k, dim=1)
        self._query_batches.append(query_ids)
        self._item_batches.append(top_item_ids)
        self._item_scores.append(top_scores)

    def get_result(self) -> _T:
        """
        :returns: prediction result
        """
        prediction = self._ids_to_result(
            torch.cat(self._query_batches),
            torch.cat(self._item_batches),
            torch.cat(self._item_scores),
        )

        return prediction

    def _compute_pipeline(
            self, query_ids: torch.LongTensor, scores: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        for postprocessor in self._postprocessors:
            query_ids, scores = postprocessor.on_prediction(query_ids, scores)
        return query_ids, scores

    @abc.abstractmethod
    def _ids_to_result(
            self,
            query_ids: torch.Tensor,
            item_ids: torch.Tensor,
            item_scores: torch.Tensor,
    ) -> _T:  # pragma: no cover
        pass


class PandasPredictionCallback(BasePredictionCallback[PandasDataFrame]):
    """
    Callback for predition stage with pandas data frame
    """

    def _ids_to_result(
            self,
            query_ids: torch.Tensor,
            item_ids: torch.Tensor,
            item_scores: torch.Tensor,
    ) -> PandasDataFrame:
        prediction = PandasDataFrame(
            {
                self.query_column: query_ids.flatten().cpu().numpy(),
                self.item_column: list(item_ids.cpu().numpy()),
                self.rating_column: list(item_scores.cpu().numpy()),
            }
        )
        return prediction.explode([self.item_column, self.rating_column])


class TorchPredictionCallback(BasePredictionCallback[Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]]):
    """
    Callback for predition stage with tuple of tensors
    """

    def __init__(
            self,
            top_k: int,
            postprocessors: Optional[List[BasePostProcessor]] = None,
    ) -> None:
        """
        :param top_k: Takes the highest k scores in the ranking.
        :param postprocessors: postprocessors to apply.
        """
        super().__init__(
            top_k=top_k,
            query_column="query_id",
            item_column="item_id",
            rating_column="rating",
            postprocessors=postprocessors,
        )

    def _ids_to_result(
            self,
            query_ids: torch.Tensor,
            item_ids: torch.Tensor,
            item_scores: torch.Tensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        return (
            cast(torch.LongTensor, query_ids.flatten().cpu().long()),
            cast(torch.LongTensor, item_ids.cpu().long()),
            item_scores.cpu(),
        )


class QueryEmbeddingsPredictionCallback(lightning.Callback):
    """
    Callback for prediction stage to get query embeddings.
    """

    def __init__(self):
        self._embeddings_per_batch: List[torch.Tensor] = []

    def on_predict_epoch_start(
            self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:
        self._embeddings_per_batch.clear()

    def on_predict_batch_end(
            self,
            trainer: lightning.Trainer,  # noqa: ARG002
            pl_module: lightning.LightningModule,
            outputs: torch.Tensor,  # noqa: ARG002
            batch: PredictionBatch,
            batch_idx: int,  # noqa: ARG002
            dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        args = [batch.features, batch.padding_mask]
        if isinstance(pl_module, Bert4Rec):
            args.append(batch.tokens_mask)

        query_embeddings = pl_module._model.get_query_embeddings(*args)
        self._embeddings_per_batch.append(query_embeddings)

    def get_result(self):
        """
        :returns: Query embeddings through all batches.
        """
        return torch.cat(self._embeddings_per_batch)



class ValidationBatch(Protocol):
    """
    Validation callback batch
    """

    query_id: torch.LongTensor
    ground_truth: torch.LongTensor
    train: torch.LongTensor


class ValidationMetricsCallback(lightning.Callback):
    """
    Callback for validation and testing stages.

    If multiple validation/testing dataloaders are used,
    the suffix of the metric name will contain the serial number of the dataloader.
    """

    def __init__(
            self,
            metrics: Optional[List[CallbackMetricName]] = None,
            ks: Optional[List[int]] = None,
            postprocessors: Optional[List[BasePostProcessor]] = None,
            item_count: Optional[int] = None,
    ):
        """
        :param metrics: Sequence of metrics to calculate.
        :param ks: highest k scores in ranking. Default: will be `[1, 5, 10, 20]`.
        :param postprocessors: postprocessors to validation stage.
        :param item_count: the total number of items in the dataset, required only for Coverage calculations.
        """
        self._metrics = metrics
        self._ks = ks
        self._item_count = item_count
        self._metrics_builders: List[TorchMetricsBuilder] = []
        self._dataloaders_size: List[int] = []
        self._postprocessors: List[BasePostProcessor] = postprocessors or []

    def _get_dataloaders_size(self, dataloaders: Optional[Any]) -> List[int]:
        if isinstance(dataloaders, torch.utils.data.DataLoader):
            return [len(dataloaders)]
        return [len(dataloader) for dataloader in dataloaders]

    def on_validation_epoch_start(
            self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:
        self._dataloaders_size = self._get_dataloaders_size(trainer.val_dataloaders)
        self._metrics_builders = [
            TorchMetricsBuilder(self._metrics, self._ks, self._item_count) for _ in self._dataloaders_size
        ]
        for builder in self._metrics_builders:
            builder.reset()

    def on_test_epoch_start(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,  # noqa: ARG002
    ) -> None:  # pragma: no cover
        self._dataloaders_size = self._get_dataloaders_size(trainer.test_dataloaders)
        self._metrics_builders = [
            TorchMetricsBuilder(self._metrics, self._ks, self._item_count) for _ in self._dataloaders_size
        ]
        for builder in self._metrics_builders:
            builder.reset()

    def _compute_pipeline(
            self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        for postprocessor in self._postprocessors:
            query_ids, scores, ground_truth = postprocessor.on_validation(query_ids, scores, ground_truth)
        return query_ids, scores, ground_truth

    def on_validation_batch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            outputs: torch.Tensor,
            batch: ValidationBatch,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
            self,
            trainer: lightning.Trainer,
            pl_module: lightning.LightningModule,
            outputs: torch.Tensor,
            batch: ValidationBatch,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:  # pragma: no cover
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _batch_end(
            self,
            trainer: lightning.Trainer,  # noqa: ARG002
            pl_module: lightning.LightningModule,
            outputs: torch.Tensor,
            batch: ValidationBatch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        _, seen_scores, seen_ground_truth = self._compute_pipeline(batch.query_id, outputs, batch.ground_truth)
        sampled_items = torch.topk(seen_scores, k=self._metrics_builders[dataloader_idx].max_k, dim=1).indices
        self._metrics_builders[dataloader_idx].add_prediction(sampled_items, seen_ground_truth, batch.train)

        if batch_idx + 1 == self._dataloaders_size[dataloader_idx]:
            pl_module.log_dict(
                self._metrics_builders[dataloader_idx].get_metrics(),
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=True,
            )

    def on_validation_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        self._epoch_end(trainer, pl_module)

    def on_test_epoch_end(
            self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:  # pragma: no cover
        self._epoch_end(trainer, pl_module)

    def _epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:  # noqa: ARG002
        @rank_zero_only
        def print_metrics() -> None:
            metrics = {}
            for name, value in trainer.logged_metrics.items():
                if "@" in name:
                    metrics[name] = value.item()

            if metrics:
                metrics_df = metrics_to_df(metrics)

                print(metrics_df)  # noqa: T201
                print()  # noqa: T201

        print_metrics()

