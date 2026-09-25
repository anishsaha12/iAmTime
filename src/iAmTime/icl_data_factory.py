from copy import deepcopy
from functools import partial
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, cast
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin
from datasets import load_dataset, Dataset
from .utils_icl import GetPaddedBatchInputsForICL, encode_categorical_exog


@dataclass
class ICLDataCollator(DataCollatorMixin):
    """
    Custom data collator for ICL (In-Context Learning) training.
    
    Handles batching of samples with the following structure:
    - example_target_histories: list of examples with time series context data for each
    - example_target_hist_mask: boolean mask for valid context data
    - example_exog_histories: list of examples with context's exogenous data for each
    - example_exog_hist_mask: boolean mask for valid context's exogenous data
    - example_target_futures: list of examples with future target values for each
    - example_target_fut_mask: boolean mask for valid future target values
    - example_exog_futures: list of examples with context's exogenous future data for each
    - example_exog_fut_mask: boolean mask for valid context's exogenous future data
    - query_target_history: query time series context data
    - query_target_hist_mask: boolean mask for valid context data
    - query_exog_history: query time series context's exogenous data
    - query_exog_hist_mask: boolean mask for valid context's exogenous data
    - query_target_future: query time series future target values (target data to compute loss)
    - query_target_fut_mask: boolean mask for valid future target values
    - query_exog_future: query time series exogenous future values
    - query_exog_fut_mask: boolean mask for valid exogenous future values
    
    Args:
        return_tensors: Type of tensors to return ('pt' for PyTorch)
    """
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.
        
        Args:
            features: List of dictionaries, each containing:
                - 'example_target_histories': tensors of shape (num_examples, hist_len)
                - 'example_target_hist_mask': tensors of shape (num_examples, hist_len)
                - 'example_exog_histories': tensors of shape (num_examples, num_exog, hist_len).
                                            Here num_exog can be different for each sample
                - 'example_exog_hist_mask': tensors of shape (num_examples, num_exog, hist_len)
                - 'example_target_futures': tensors of shape (num_examples, future_len)
                - 'example_target_fut_mask': tensors of shape (num_examples, future_len)
                - 'example_exog_futures': tensors of shape (num_examples, num_exog, future_len).
                                          Here num_exog can be different for each sample
                - 'example_exog_fut_mask': tensors of shape (num_examples, num_exog, future_len)
                - 'query_target_history': tensors of shape (query_hist_len)
                - 'query_target_hist_mask': tensors of shape (query_hist_len)
                - 'query_exog_history': tensors of shape (num_exog, query_hist_len).
                                        Here num_exog can be different for each sample
                - 'query_exog_hist_mask': tensors of shape (num_exog, query_hist_len)
                - 'query_target_future': tensors of shape (future_len) for loss
                - 'query_target_fut_mask': tensors of shape (future_len) for loss
                - 'query_exog_future': tensors of shape (num_exog, future_len).
                                       Here num_exog can be different for each sample
                - 'query_exog_fut_mask': tensors of shape (num_exog, future_len)
                - 'start': datetime representing start of 'query_target_history'
                - 'item_id': ID of 'query_target'
                - 'exog_item_ids': List of IDs of length num_exog for each 'query_target'.
                - 'example_starts': List of datetimes representing start of each example's 
                                    'example_target_histories' in the batch.

        Returns:
            Batched dictionary with tensors of shapes (batch_size, num_examples, num_exog, sequence_length), 
            (batch_size, num_exog, sequence_length) and (batch_size, sequence_length)
        """
        
        # Stack tensors along batch dimension
        icl_batch = GetPaddedBatchInputsForICL(
            model_icl_inputs=features
        )
        icl_batch_stacked = icl_batch.get_padded_batch()
        icl_batch_masks_stacked = icl_batch.get_padded_batch_masks_from_measurements(
            model_icl_input_masks=features
        )

        return {
            ## inputs
            "example_target_histories": icl_batch_stacked["example_target_histories"],
            "example_target_futures": icl_batch_stacked["example_target_futures"],
            "example_exog_histories": icl_batch_stacked["example_exog_histories"],
            "example_exog_futures": icl_batch_stacked["example_exog_futures"],
            "query_target_history": icl_batch_stacked["query_target_history"],
            "query_exog_history": icl_batch_stacked["query_exog_history"],
            "query_exog_future": icl_batch_stacked["query_exog_future"],
            "target": icl_batch_stacked["query_target_future"],
            ## masks
            "example_target_hist_mask": icl_batch_masks_stacked["example_target_hist_mask"],
            "example_target_fut_mask": icl_batch_masks_stacked["example_target_fut_mask"],
            "example_exog_hist_mask": icl_batch_masks_stacked["example_exog_hist_mask"],
            "example_exog_fut_mask": icl_batch_masks_stacked["example_exog_fut_mask"],
            "query_target_hist_mask": icl_batch_masks_stacked["query_target_hist_mask"],
            "query_exog_hist_mask": icl_batch_masks_stacked["query_exog_hist_mask"],
            "query_exog_fut_mask": icl_batch_masks_stacked["query_exog_fut_mask"],
            "target_mask": icl_batch_masks_stacked["query_target_fut_mask"],
        }


def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class CSVTimeSeriesDataset:
    """
    A dataset class for loading time series data from CSV files.
    
    Supports two CSV formats:
    1. Long format:
       - unique_id: identifier for the time series
       - ds: datetime column
       - y: target values
    2. Wide format:
       - date: datetime column
       - multiple columns (0, 1, 2, ...) representing different time series
    """
    
    def __init__(self, csv_path: str, exog_rel_path: str = None, freq: str = "D", format_type: str = "auto"):
        self.csv_path = Path(csv_path)
        self.exog_rel_path = Path(exog_rel_path) if exog_rel_path else None
        self.freq = freq
        self.format_type = format_type
        self._data = None
        self._load_data()
    
    def _detect_format(self, df):
        """Detect whether CSV is in long or wide format."""
        if 'unique_id' in df.columns and 'ds' in df.columns and 'y' in df.columns:
            return 'long'
        elif 'date' in df.columns and len(df.columns) > 2:
            return 'wide'
        else:
            raise ValueError(
                "Cannot detect CSV format. Expected either long format (unique_id, ds, y)\
                      or wide format (date, col1, col2, ...)"
            )

    def _load_data(self):
        """
        Load and preprocess the CSV data. If exogenous relations are provided, use that. 
        Else, each time series in the CSV is considered to be the target series.
        """
        df = pd.read_csv(self.csv_path)
        exog_rel = json.loads(self.exog_rel_path.read_text()) if self.exog_rel_path else None

        # Auto-detect format if not specified
        if self.format_type == "auto":
            self.format_type = self._detect_format(df)
        
        if self.format_type == "long":
            self._load_long_format(df, exog_rel)
        elif self.format_type == "wide":
            self._load_wide_format(df, exog_rel)
        else:
            raise ValueError(f"Unknown format_type: {self.format_type}")

    def _load_long_format(self, df: pd.DataFrame, exog_rel: list[dict]=None):
        """Load data in long format (unique_id, ds, y)."""
        # Ensure required columns exist
        required_cols = ['unique_id', 'ds', 'y']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Convert ds to datetime
        df['ds'] = pd.to_datetime(df['ds'])
        unique_ids = df['unique_id'].unique().tolist()

        if exog_rel:
            all_rel_ids = [exr['target'] for exr in exog_rel]
            for exr in exog_rel:
                all_rel_ids.extend(exr['exogenous'])
            all_rel_ids = set(all_rel_ids)
            missing_unique_ids = set(unique_ids) - all_rel_ids

            self._data = []
            for rel in exog_rel:
                rel_df = df.loc[
                    df['unique_id'].isin([rel['target']] + rel['exogenous'])
                ].pivot(index='ds', columns='unique_id', values='y').reset_index().sort_values(
                    by='ds'
                ).reset_index(drop=True)
                target_ts = rel_df[rel['target']].values.astype(np.float32)
                exog_ts = rel_df[rel['exogenous']].values.T.astype(np.float32)
                entry = {
                    'start': rel_df['ds'].iloc[0],
                    'target': target_ts,
                    'exogenous': exog_ts,
                    'item_id': str(rel['target']),
                    'exog_item_ids': rel['exogenous']
                }
                self._data.append(entry)

            missing_df = df.loc[
                df['unique_id'].isin(missing_unique_ids)
            ].sort_values(['unique_id', 'ds']).reset_index(drop=True)
            for unique_id, group in missing_df.groupby('unique_id'):
                target_ts = group['y'].values.astype(np.float32)
                entry = {
                    'start': group['ds'].iloc[0],
                    'target': target_ts,
                    'exogenous': np.empty((0,target_ts.shape[0])),
                    'item_id': str(unique_id),
                    'exog_item_ids': [],
                }
                self._data.append(entry)
        else:
            # Sort by unique_id and ds
            df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
            
            # Group by unique_id to create separate time series
            self._data = []
            for unique_id, group in df.groupby('unique_id'):
                # Create entry in GluonTS format for compatibility
                target_ts = group['y'].values.astype(np.float32)
                entry = {
                    'start': group['ds'].iloc[0],  # First timestamp
                    'target': target_ts,
                    'exogenous': np.empty((0,target_ts.shape[0])),
                    'item_id': str(unique_id),
                    'exog_item_ids': [],
                }
                self._data.append(entry)
    
    def _load_wide_format(self, df, exog_rel: dict=None):
        """Load data in wide format (date, col1, col2, ...)."""
        # Ensure date column exists
        if 'date' not in df.columns:
            raise ValueError("Required column 'date' not found in wide format CSV")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Get all columns except date as time series columns
        ts_columns = [col for col in df.columns if col != 'date']

        if exog_rel:
            all_rel_ids = [exr['target'] for exr in exog_rel]
            for exr in exog_rel:
                all_rel_ids.extend(exr['exogenous'])
            all_rel_ids = set(all_rel_ids)
            missing_unique_ids = set(ts_columns) - all_rel_ids

            # Create separate time series for each exogenous variable
            self._data = []
            for rel in exog_rel:
                target_ts = df[rel['target']].values.astype(np.float32)
                target_ts[target_ts == 0.001] = np.nan
                exog_ts = df[rel['exogenous']].values.T.astype(np.float32)
                exog_ts[exog_ts == 0.001] = np.nan
                entry = {
                    'start': df['date'].iloc[0],  # First timestamp
                    'target': target_ts,
                    'exogenous': exog_ts,
                    'item_id': str(rel['target']),
                    'exog_item_ids': rel['exogenous']
                }
                self._data.append(entry)

            for col in missing_unique_ids:
                # Replace small values (0.001) with NaN as they seem to be placeholders
                target_values = df[col].values.astype(np.float32)
                target_values[target_values == 0.001] = np.nan
                
                entry = {
                    'start': df['date'].iloc[0],  # First timestamp
                    'target': target_values,
                    'exogenous': np.empty((0,target_values.shape[0])),
                    'item_id': str(col),
                    'exog_item_ids': []
                }
                self._data.append(entry)

        else:
            # Create separate time series for each column
            self._data = []
            for col in ts_columns:
                # Replace small values (0.001) with NaN as they seem to be placeholders
                target_values = df[col].values.astype(np.float32)
                target_values[target_values == 0.001] = np.nan
                
                entry = {
                    'start': df['date'].iloc[0],  # First timestamp
                    'target': target_values,
                    'exogenous': np.empty((0,target_values.shape[0])),
                    'item_id': str(col),
                    'exog_item_ids': []
                }
                self._data.append(entry)
    
    def __iter__(self):
        """Iterate over the time series entries."""
        for entry in self._data:
            yield entry
    
    def __len__(self):
        return len(self._data)


class Filter:
    """Filter dataset entries based on a condition function."""
    
    def __init__(self, condition_fn, dataset):
        self.condition_fn = condition_fn
        self.dataset = dataset
    
    def __iter__(self):
        for entry in self.dataset:
            if self.condition_fn(entry):
                yield entry


class Cyclic:
    """Create a cyclic iterator that repeats indefinitely."""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __iter__(self):
        while True:
            for entry in self.dataset:
                yield entry


class Map:
    """Apply a transformation function to each entry."""
    
    def __init__(self, transform_fn, dataset):
        self.transform_fn = transform_fn
        self.dataset = dataset
    
    def __iter__(self):
        for entry in self.dataset:
            yield self.transform_fn(entry)


class InstanceSplitter:
    """Split time series into training instances with past and future windows."""
    
    def __init__(
        self,
        target_field: str = "target",
        past_length: int = 512,
        future_length: int = 64,
        instance_sampler = None,
        dummy_value: float = np.nan,
        **kwargs
    ):
        self.target_field = target_field
        self.past_length = past_length
        self.future_length = future_length
        self.instance_sampler = instance_sampler
        self.dummy_value = dummy_value
    
    def apply(self, dataset, is_train: bool = True):
        """Apply the instance splitting transformation."""
        return InstanceSplitDataset(
            dataset, 
            self.past_length, 
            self.future_length,
            self.instance_sampler,
            self.dummy_value,
            is_train,
        )


class InstanceSplitDataset:
    """Dataset that yields past/future splits of time series."""
    
    def __init__(self, dataset, past_length: int, future_length: int, 
                 instance_sampler=None, dummy_value: float = np.nan,
                 is_train: bool = True):
        self.dataset = dataset
        self.past_length = past_length
        self.future_length = future_length
        self.instance_sampler = instance_sampler
        self.dummy_value = dummy_value
        self.is_train = is_train

    def pad_axis(
        self,
        a: np.ndarray, *, axis: int = 0, left: int = 0, right: int = 0, value=0
    ) -> np.ndarray:
        """Similar to ``np.pad``, but pads only a single axis using `left` and
        `right` parameters.
        ::
            >>> pad_axis([1, 2, 3, 4], left=2, right=3)
            array([0, 0, 1, 2, 3, 4, 0, 0, 0])
        """
        a = np.array(a)

        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (left, right)
        return np.pad(a, pad_width, constant_values=value)

    def _split_array(
        self, array: np.ndarray, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if idx >= self.past_length:
            past_piece = array[..., idx - self.past_length : idx]
        else:
            past_piece = array[..., :idx] # Dont pad yet as padding is done in collator
            # self.pad_axis(
            #     array[..., :idx],
            #     axis=-1,
            #     left=self.past_length - idx,
            #     value=self.dummy_value,
            # )

        future_start = idx
        future_slice = slice(future_start, future_start + self.future_length)
        future_piece = array[..., future_slice]

        return past_piece, future_piece

    def _split_instance(self, entry, idx: int):
        past_piece, future_piece = self._split_array(entry["target"], idx)
        past_exog_piece, future_exog_piece = self._split_array(entry["exogenous"], idx)
        try:
            start_time = entry.get("start") + pd.Timedelta(days=idx)
        except:
            start_time = entry.get("start")
        return {
            "past_target": past_piece,
            "future_target": future_piece,
            "past_exogenous": past_exog_piece,
            "future_exogenous": future_exog_piece,
            "start": start_time,
            "item_id": entry.get("item_id"),
            "exog_item_ids": entry.get("exog_item_ids"),
        }
    
    def __iter__(self):
        for entry in self.dataset:
            sampled_indices = self.instance_sampler(entry["target"])
            for idx in sampled_indices:
                yield self._split_instance(entry, idx)


class ExpectedNumInstanceSampler:
    """
    Sampler for training that controls the number of instances per time series.
    The sampled indices ``i`` satisfy ``a <= i <= b``, where ``a = min_past``
    and ``b = ts.shape[-1] - min_future``.
    """
    total_length: int = 0
    n: int = 0
    
    def __init__(self, num_instances: float = 1.0, min_instances: int = 1, 
                 min_past: int = 1, min_future: int = 1):
        self.num_instances = num_instances
        self.min_instances = min_instances
        self.min_past = min_past
        self.min_future = min_future

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[-1] - self.min_future,
        )
    
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1

        if window_size <= 0:
            return np.array([], dtype=int)

        self.n += 1
        self.total_length += window_size
        avg_length = self.total_length / self.n

        if avg_length <= 0:
            return np.array([], dtype=int)

        p = self.num_instances / avg_length
        (indices,) = np.where(np.random.random_sample(window_size) < p)
        indices += a
        if len(indices) < self.min_instances:
            prefix = np.random.randint(
                a, b + 1, size=self.min_instances - len(indices)
            )
            return np.concatenate([prefix, indices])

        return indices


class TestSplitSampler:
    """Sampler for test data."""
    def __init__(self, min_past: int = 0):
        self.min_past = min_past
        self.min_future = 0

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[-1] - self.min_future,
        )
    
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        return np.array([b]) if a <= b else np.array([], dtype=int)


class ValidationSplitSampler:
    """Sampler for validation data."""
    
    def __init__(self, min_past: int = 0, min_future: int = 0):
        self.min_past = min_past
        self.min_future = min_future
    
    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[-1] - self.min_future,
        )
    
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        return np.array([b]) if a <= b else np.array([], dtype=int)


class LeavesMissingValues:
    """Imputation method that leaves missing values as-is."""
    
    def __call__(self, target):
        return target


class LastValueImputation:
    """Imputation method that forward-fills missing values."""
    
    def __call__(self, target):
        target = target.copy()
        mask = np.isnan(target)
        if mask.any():
            # Forward fill
            for i in range(1, len(target)):
                if mask[i] and not mask[i-1]:
                    target[i] = target[i-1]
        return target


class iAmTimeDataset(IterableDataset, ShuffleMixin):
    """
    An iterable PyTorch dataset for iAmTime time series forecasting.

    This dataset wrapper yields dictionaries containing 'context' and 'mask' tensors,
    suitable for training, validation, and testing with iAmTime models. It supports
    instance splitting, imputation of missing values, probabilistic sampling from multiple
    datasets, and optional value dropping for training regularization.

    Args:
        datasets (list): List of input datasets, each providing time series entries.
        probabilities (list): Sampling probabilities for each dataset.
        context_length (int, optional): Number of past time steps to use as context. Default is 512.
        prediction_length (int, optional): Number of future time steps to predict. Default is 64.
        drop_prob (float, optional): Probability of randomly dropping values in the target during training. 
                                     Default is 0.0.
        min_past (int, optional): Minimum required past context length. 
                                  Defaults to prediction_length if not specified.
        imputation_method (callable, optional): Method to impute missing values in the target. 
                                                Defaults to LeavesMissingValues().
        mode (str, optional): Mode of operation, one of "training", "validation", or "test". 
                              Default is "training".
        np_dtype (np.dtype, optional): Numpy data type for target arrays. Default is np.float32.
        min_past_context_std (float, optional): Minimum standard deviation for past context. Default is 1e-6.

    Methods:
        preprocess_entry(entry, mode): Preprocesses a single entry, applies imputation 
                                       and optional value dropping.
        _create_instance_splitter(mode): Creates an instance splitter for the specified mode.
        create_training_data(data): Applies training-specific transformations and filtering.
        create_test_data(data): Applies test-specific transformations.
        create_validation_data(data): Applies validation-specific transformations.
        to_hf_format(entry): Converts an entry to HuggingFace-compatible tensor format.
        __iter__(): Iterates over the dataset, yielding formatted entries according to the mode.

    Yields:
        dict: A dictionary with keys 'context', 'mask', 'target', and 'target_mask', each a torch tensor.
    """
    def __init__(
        self,
        datasets,
        probabilities,
        context_length=512,
        prediction_length=64,
        drop_prob: float = 0.0,
        min_past=None,
        imputation_method=None,
        mode="training",
        np_dtype=np.float32,
        min_past_context_std=1e-6
    ):
        super().__init__()
        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        self.datasets = datasets
        self.probabilities = probabilities
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob
        self.min_past = min_past or prediction_length
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype
        self.min_past_context_std = min_past_context_std

    def preprocess_entry(self, entry, mode):
        entry = {
            f: entry[f] 
            for f in ['start', 'target', 'exogenous', 'item_id', 'exog_item_ids']
            if f in entry
        }
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        entry["exogenous"] = np.asarray(entry["exogenous"], dtype=self.np_dtype)
        if entry["exogenous"].size == 0:
            entry["exogenous"] = np.empty((0, entry["target"].shape[0]), dtype=self.np_dtype)
        assert entry["target"].ndim == 1
        assert entry["exogenous"].ndim == 2
        # Impute if needed
        entry["target"] = self.imputation_method(entry["target"])
        # Optionally drop values for training
        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target
        return entry

    def _create_instance_splitter(self, mode):
        assert mode in ["training", "test", "validation"]
        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]
        return InstanceSplitter(
            target_field="target",
            past_length=self.context_length,
            future_length=self.prediction_length,
            instance_sampler=instance_sampler,
            dummy_value=np.nan,
        )
    
    def normalize_context(self, x, eps=1e-5, keepdims=True, return_normalized=False, loc_scale=None):
        if (loc_scale is not None):
            loc, scale = loc_scale
        else:
            loc = np.nanmean(x, axis=-1, keepdims=keepdims)
            loc = np.nan_to_num(loc, nan=0.0)
            scale = np.nanmean((x - loc) ** 2, axis=-1, keepdims=keepdims)
            scale = np.sqrt(scale)
            scale = np.nan_to_num(scale, nan=1.0)
        if return_normalized==True:
            clipped_scale = np.where(scale == 0, eps, scale)
            normalized = (x - loc) / clipped_scale
            return normalized, (loc, scale)
        return (loc, scale)

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter("training")
        data = split_transform.apply(data, is_train=True)
        
        # Filter out entries with no valid past data and near-zero std
        def valid_context(entry):
            past = entry["past_target"]
            valid_count = (~np.isnan(past)).sum()
            # Exclude if all values are nan or std is near zero (e.g., < 1e-6)
            if valid_count == 0:
                return False
            loc, scale = self.normalize_context(past, keepdims=False, return_normalized=False)
            past_context_std = np.abs(np.nan_to_num(
                np.nan_to_num(scale, nan=0.0) / np.nan_to_num(loc, nan=0.0),
                nan=0.0
            ))
            if past_context_std < self.min_past_context_std:
                return False
            return True
        
        data = Filter(valid_context, data)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry, mode):
        context = entry["past_target"]
        context_exog = entry["past_exogenous"]
        future_target = entry["future_target"]
        future_target_exog = entry["future_exogenous"]

        if mode == "training":
            context, loc_scale = self.normalize_context(context, keepdims=True, return_normalized=True)
            future_target, _ = self.normalize_context(
                future_target, keepdims=True, return_normalized=True, loc_scale=loc_scale
            )

        context = torch.tensor(context).unsqueeze(0)
        mask = ~torch.isnan(context)

        context_exog = torch.tensor(context_exog).unsqueeze(0)
        mask_exog = ~torch.isnan(context_exog)

        future_target = torch.tensor(future_target).unsqueeze(0)
        future_mask = ~torch.isnan(future_target)

        future_target_exog = torch.tensor(future_target_exog).unsqueeze(0)
        future_mask_exog = ~torch.isnan(future_target_exog)

        return {
            "context": context.squeeze(0),
            "mask": mask.squeeze(0),
            "context_exog": context_exog.squeeze(0),
            "mask_exog": mask_exog.squeeze(0),
            "target": future_target.squeeze(0),
            "target_mask": future_mask.squeeze(0),
            "target_exog": future_target_exog.squeeze(0),
            "target_mask_exog": future_mask_exog.squeeze(0),
            "start": entry["start"],
            "item_id": entry["item_id"],
            "exog_item_ids": entry["exog_item_ids"],
        }

    def __iter__(self):
        preprocessed_datasets = [
            Map(partial(self.preprocess_entry, mode=self.mode), dataset)
            for dataset in self.datasets
        ]
        if self.mode == "training":
            iterables = [self.create_training_data(dataset) for dataset in preprocessed_datasets]
        elif self.mode == "test":
            iterables = [self.create_test_data(dataset) for dataset in preprocessed_datasets]
        else:
            iterables = [self.create_validation_data(dataset) for dataset in preprocessed_datasets]
        
        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(itertools.islice(self.probabilities, worker_id, None, num_workers))
        
        probs = [prob / sum(probs) for prob in probs]
        iterators = list(map(iter, iterables))
        
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs) # index of which dataset to take from
                try:
                    yield self.to_hf_format(next(iterators[idx]), mode="training")
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry, mode=self.mode)


class ExampleSplitDataset:
    """Dataset that returns examples from past/future splits of time series."""

    def __init__(self, past_length: int, future_length: int,
                 instance_sampler=None, is_train: bool = True):
        self.past_length = past_length
        self.future_length = future_length
        self.instance_sampler = instance_sampler
        self.is_train = is_train

    def _split_array(
        self, array: np.ndarray, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if idx >= self.past_length:
            past_piece = array[..., idx - self.past_length : idx]
        else:
            past_piece = array[..., :idx]

        future_start = idx
        future_slice = slice(future_start, future_start + self.future_length)
        future_piece = array[..., future_slice]

        return past_piece, future_piece

    def _split_instance(self, entry, idx: int):
        past_piece, future_piece = self._split_array(entry["target"], idx)
        past_exog_piece, future_exog_piece = self._split_array(entry["exogenous"], idx)
        try:
            start_time = entry.get("start") + pd.Timedelta(days=idx)
        except:
            start_time = entry.get("start")
        return {
            "past_target": past_piece,
            "future_target": future_piece,
            "past_exogenous": past_exog_piece,
            "future_exogenous": future_exog_piece,
            "start": start_time,
        }
    
    def sample_examples(self, entry):
        sampled_indices = self.instance_sampler(entry["target"])
        examples = []
        for idx in sampled_indices:
            examples.append(self._split_instance(entry, idx))
        return examples


class ExampleInstanceSampler:
    """
    Sampler for training that creates examples from a time-series.
    The sampled indices ``i`` satisfy ``a <= i <= b``, where ``a = min_past``
    and ``b = ts.shape[-1] - min_future``.
    """
    def __init__(self, min_instances: int = 1, max_instances: int = 1,
                 min_past: int = 1, min_future: int = 1):
        self.max_instances = max_instances
        self.min_instances = min_instances
        self.min_past = min_past
        self.min_future = min_future

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[-1] - self.min_future,
        )
    
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)

        window_size = b - a + 1
        if window_size <= 0:
            return np.array([], dtype=int)

        num_instances = np.random.randint(self.min_instances, self.max_instances + 1)
        indices = np.random.randint(a, b + 1, size=num_instances)
        if len(indices) < self.min_instances:
            prefix = np.random.randint(
                a, b + 1, size=self.min_instances - len(indices)
            )
            return np.concatenate([prefix, indices])
        return indices


class ICLDatasetForForecasting(IterableDataset):
    """
    Iterable PyTorch dataset that wraps another dataset (e.g., iAmTimeDataset).
    The base dataset is to provide time series entries containing `context` & `target` (and `exogenous` for each).
    It splits each entry into examples using ExampleSplitDataset and ExampleInstanceSampler,
    and formats each entry into the ICL format for forecasting tasks.
    """
    def __init__(
            self, base_dataset,
            min_examples=1, max_examples=1, 
            min_past_for_examples=1,
        ):
        """
        Args:
            base_dataset: Iterable PyTorch dataset - iAmTimeDataset or PseudoShuffledIterableDataset 
                          containing context_length, prediction_length, min_past, mode
            min_examples: Minimum number of examples to sample per entry
            max_examples: Maximum number of examples to sample per entry
            min_past_for_examples: Minimum required past context length for examples
            is_train: Whether to use training sampling logic
        """
        base_dataset_with_attributes = base_dataset
        while (
            base_dataset_with_attributes is not None and 
            not self._check_attributes_exist_in_base(base_dataset_with_attributes)
        ):
            base_dataset_with_attributes = getattr(base_dataset_with_attributes, "base_dataset", None)
        assert base_dataset_with_attributes is not None, "Base dataset is not valid"
        assert base_dataset_with_attributes.min_past >= min_past_for_examples, \
            "min_past_for_examples should be less than or equal to base_dataset.min_past"
        super().__init__()
        self.base_dataset = base_dataset
        self.past_length = base_dataset_with_attributes.context_length
        self.future_length = base_dataset_with_attributes.prediction_length
        self.max_examples = max_examples
        self.min_examples = min_examples
        self.min_past_for_examples = min_past_for_examples
        self.min_future_for_examples = base_dataset_with_attributes.prediction_length
        self.mode = base_dataset_with_attributes.mode

        instance_sampler = ExampleInstanceSampler(
            min_instances=self.min_examples,
            max_instances=self.max_examples,
            min_past=self.min_past_for_examples,
            min_future=self.min_future_for_examples,
        )
        if self.mode == "training":
            self.example_splitter = self.create_training_example_splitter(instance_sampler)
        elif self.mode == "validation":
            self.example_splitter = self.create_validation_example_splitter(instance_sampler)

    def _check_attributes_exist_in_base(self, base_dataset):
        return all(
            hasattr(base_dataset, attr) 
            for attr in ["context_length", "prediction_length", "min_past", "mode"]
        )

    def create_training_example_splitter(self, instance_sampler):
        return ExampleSplitDataset(
            past_length=self.past_length,
            future_length=self.future_length,
            instance_sampler=instance_sampler,
            is_train=True,
        )

    def create_validation_example_splitter(self, instance_sampler):
        return ExampleSplitDataset(
            past_length=self.past_length,
            future_length=self.future_length,
            instance_sampler=instance_sampler,
            is_train=False,
        )

    def to_icl_format(self, entry):
        context = torch.tensor(entry["context"]).unsqueeze(0)
        mask = torch.tensor(entry["mask"]).unsqueeze(0)

        context_exog = torch.tensor(entry["context_exog"]).unsqueeze(0)
        mask_exog = torch.tensor(entry["mask_exog"]).unsqueeze(0)

        future_target = torch.tensor(entry["target"]).unsqueeze(0)
        future_mask = torch.tensor(entry["target_mask"]).unsqueeze(0)

        future_target_exog = torch.tensor(entry["target_exog"]).unsqueeze(0)
        future_mask_exog = torch.tensor(entry["target_mask_exog"]).unsqueeze(0)

        ## Create examples from context and context_exog to pass to ICL model
        examples = self.example_splitter.sample_examples(
            entry={
                "target": entry["context"],
                "exogenous": entry["context_exog"],
                "start": entry["start"],
            }
        )

        # Format examples into tensors
        example_start = []
        example_target_histories = []
        example_target_hist_mask = []
        example_exog_histories = []
        example_exog_hist_mask = []
        example_target_futures = []
        example_target_fut_mask = []
        example_exog_futures = []
        example_exog_fut_mask = []

        for ex in examples:
            t_hist = torch.tensor(ex["past_target"]).unsqueeze(0)
            t_hist_mask = ~torch.isnan(t_hist)
            exog_hist = torch.tensor(ex["past_exogenous"]).unsqueeze(0)
            exog_hist_mask = ~torch.isnan(exog_hist)
            t_fut = torch.tensor(ex["future_target"]).unsqueeze(0)
            t_fut_mask = ~torch.isnan(t_fut)
            exog_fut = torch.tensor(ex["future_exogenous"]).unsqueeze(0)
            exog_fut_mask = ~torch.isnan(exog_fut)

            example_target_histories.append(t_hist.squeeze(0))
            example_target_hist_mask.append(t_hist_mask.squeeze(0))
            example_exog_histories.append(exog_hist.squeeze(0))
            example_exog_hist_mask.append(exog_hist_mask.squeeze(0))
            example_target_futures.append(t_fut.squeeze(0))
            example_target_fut_mask.append(t_fut_mask.squeeze(0))
            example_exog_futures.append(exog_fut.squeeze(0))
            example_exog_fut_mask.append(exog_fut_mask.squeeze(0))
            example_start.append(ex["start"])

        return {
            "example_target_histories" : example_target_histories,   # (num_examples, hist_len)
            "example_target_hist_mask" : example_target_hist_mask,   # (num_examples, hist_len)

            "example_exog_histories" : example_exog_histories,       # (num_examples, num_exog, hist_len)
            "example_exog_hist_mask" : example_exog_hist_mask,       # (num_examples, num_exog, hist_len)

            "example_target_futures" : example_target_futures,       # (num_examples, future_len)
            "example_target_fut_mask" : example_target_fut_mask,     # (num_examples, future_len)

            "example_exog_futures" : example_exog_futures,           # (num_examples, num_exog, future_len)
            "example_exog_fut_mask" : example_exog_fut_mask,         # (num_examples, num_exog, future_len)

            "query_target_history" : context.squeeze(0),             # (query_hist_len)
            "query_target_hist_mask" : mask.squeeze(0),              # (query_hist_len)

            "query_exog_history" : context_exog.squeeze(0),          # (num_exog, query_hist_len)
            "query_exog_hist_mask" : mask_exog.squeeze(0),           # (num_exog, query_hist_len)

            "query_target_future" : future_target.squeeze(0),        # (future_len) for loss
            "query_target_fut_mask" : future_mask.squeeze(0),        # (future_len) for loss

            "query_exog_future" : future_target_exog.squeeze(0),     # (num_exog, future_len)
            "query_exog_fut_mask" : future_mask_exog.squeeze(0),     # (num_exog, future_len)

            "start": entry["start"],
            "item_id": entry["item_id"],
            "exog_item_ids": entry["exog_item_ids"],
            "example_starts": example_start,                         # (num_examples)
        }

    def __iter__(self):
        for entry in self.base_dataset:
            # Format entry for ICL
            yield self.to_icl_format(entry)


def trim_entry_by_validation_offset(entry: dict, offset: int, prediction_length: int):
    """
    For each entry, trim it using offset and prediction_length.

    Args:
        entry: A dictionary containing the time series data.
        offset: Integer, can be negative. Start index for the forecast window (relative to end).
        prediction_length: Number of steps to forecast.

    Returns:
        dict: entry dict has trimmed 'target' and 'exogenous' series.
    """
    if (offset is not None) and (prediction_length is not None) and (offset != -prediction_length):
        entry = deepcopy(entry)
        if "target" in entry:
            n = len(entry["target"])
            # Calculate start index for forecast window
            start = n + offset if offset < 0 else offset
            end = start + prediction_length
            # Ensure indices are valid
            if start < 0 or end > n:
                return entry  # skip if not enough data
            entry["target"] = entry["target"][:end]
            if isinstance(entry["target"], list):
                entry["target"] = np.array(entry["target"])
        if "exogenous" in entry and isinstance(entry["exogenous"], (np.ndarray, list)) and len(entry["exogenous"]):
            exog = np.array(entry["exogenous"])
            entry["exogenous"] = exog[:, :end]
        else:
            entry["exogenous"] = np.empty((0, entry["target"].shape[0]), dtype=np.float32)
    return entry

def get_base_dataset(
        data_paths: List[str],
        exogenous_relations_paths: List[str],
        probability: List[float],
        max_missing_prop: float,
        shuffle_buffer_length: int,
        min_past: int,
        min_past_context_std: float,
        context_length: int = 2048,
        prediction_length: int = 64,
        mode: str = "train",
        local_cache_dir: str = ".",
        validation_offset: int = None,
    ) -> IterableDataset:
    """
    Create a dataset for training iAmTime models.
    
    Args:
        data_paths: List of paths to CSV files (supports both long and wide formats)
        probability: Sampling probabilities for each dataset
        max_missing_prop: Maximum proportion of missing values allowed
        shuffle_buffer_length: Size of shuffle buffer
        min_past: Minimum past context length required
        context_length: Length of context window
        prediction_length: Length of prediction window
        mode: Dataset mode ("train", "test", "validation")
        local_cache_dir: Directory for caching data

    CSV Format Support:
        - Long format: columns (unique_id, ds, y)
        - Wide format: columns (date, col1, col2, ...) where each column is a time series
    
    Returns:
        IterableDataset: Dataset ready for training
    """
    if mode == "train":
        data_loader_function = {
            "parquet": lambda data_path, exog_rel_path: load_dataset(
                "parquet", 
                data_files=data_path, 
                cache_dir=local_cache_dir, 
                keep_in_memory=False
            )["train"],
            "csv": lambda data_path, exog_rel_path: CSVTimeSeriesDataset(
                csv_path=data_path,
                exog_rel_path=exog_rel_path,
                freq="D"
            )
        }
        train_datasets = [
            Filter(
                partial(
                    has_enough_observations,
                    min_length=min_past + prediction_length,
                    max_missing_prop=max_missing_prop,
                ),
                data_loader_function[data_path.split('.')[-1].lower()](
                    data_path=data_path,
                    exog_rel_path=exog_rel_path
                )
            )
            for (data_path, exog_rel_path) in 
            zip(data_paths, exogenous_relations_paths)
        ]

        shuffled_train_dataset = iAmTimeDataset(
            datasets=train_datasets,
            probabilities=probability,
            context_length=context_length,
            prediction_length=prediction_length,
            min_past=min_past,
            min_past_context_std=min_past_context_std,
            imputation_method=None,
            mode="training",
        ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

        return shuffled_train_dataset
    elif mode == "validation":
        data_loader_function = {
            "parquet": lambda data_path, exog_rel_path: load_dataset(
                "parquet", 
                data_files=data_path, 
                cache_dir=local_cache_dir, 
                keep_in_memory=False
            )["train"],
            "csv": lambda data_path, exog_rel_path: CSVTimeSeriesDataset(
                csv_path=data_path,
                exog_rel_path=exog_rel_path,
                freq="D"
            )
        }
        val_datasets = [
            Map(
                partial(
                    trim_entry_by_validation_offset, 
                    offset=validation_offset, 
                    prediction_length=prediction_length
                ), 
                data_loader_function[data_path.split('.')[-1].lower()](
                    data_path=data_path,
                    exog_rel_path=exog_rel_path
                )
            )
            for (data_path, exog_rel_path) in 
            zip(data_paths, exogenous_relations_paths)
        ]

        validation_dataset = iAmTimeDataset(
            datasets=val_datasets,
            probabilities=probability,
            context_length=context_length,
            prediction_length=prediction_length,
            min_past=min_past, # not relevant for validation
            imputation_method=None,
            mode="validation",
        )

        return validation_dataset
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
def get_icl_dataset(
        data_paths: List[str],
        exogenous_relations_paths: List[str],
        probability: List[float],
        max_missing_prop: float,
        shuffle_buffer_length: int,
        min_past: int,
        min_past_context_std: float,
        context_length: int = 2048,
        prediction_length: int = 64,
        min_examples: int = 0,
        max_examples: int = 1,
        min_past_for_examples: int = 128,
        mode: str = "train",
        local_cache_dir: str = ".",
        validation_offset: int = None
    ) -> IterableDataset:
    """
    Create a dataset for training iAmTimeModelForICL models.
    
    Args:
        data_paths: List of paths to parquet OR CSV files (supports both long and wide formats)
        probability: Sampling probabilities for each dataset
        max_missing_prop: Maximum proportion of missing values allowed
        shuffle_buffer_length: Size of shuffle buffer
        min_past: Minimum past context length required
        context_length: Length of context window
        prediction_length: Length of prediction window
        min_examples: Minimum number of examples required
        max_examples: Maximum number of examples allowed
        min_past_for_examples: Minimum past context length for examples
        mode: Dataset mode ("train", "test", "validation")
        local_cache_dir: Directory for caching datasets
        validation_offset: Offset to trim validation dataset

    CSV Format Support:
        - Long format: columns (unique_id, ds, y)
        - Wide format: columns (date, col1, col2, ...) where each column is a time series
    
    Returns:
        IterableDataset: ICL Dataset ready for training
    """
    if mode == "train":
        shuffled_train_dataset = get_base_dataset(
            data_paths=data_paths,
            exogenous_relations_paths=exogenous_relations_paths,
            probability=probability,
            max_missing_prop=max_missing_prop,
            shuffle_buffer_length=shuffle_buffer_length,
            min_past=min_past,
            min_past_context_std=min_past_context_std,
            context_length=context_length,
            prediction_length=prediction_length,
            mode=mode,
            local_cache_dir=local_cache_dir,
        )
        icl_train_dataset = ICLDatasetForForecasting(
            base_dataset=shuffled_train_dataset,
            min_examples=min_examples,
            max_examples=max_examples,
            min_past_for_examples=min_past_for_examples,
        )

        return icl_train_dataset
    if mode == "validation":
        validation_dataset = get_base_dataset(
            data_paths=data_paths,
            exogenous_relations_paths=exogenous_relations_paths,
            probability=[1. for _ in range(len(data_paths))],
            max_missing_prop=None,
            shuffle_buffer_length=None,
            min_past=max(prediction_length, min_past_for_examples), # is not relevant for validation
            min_past_context_std=min_past_context_std, # not relevant for validation
            context_length=context_length,
            prediction_length=prediction_length,
            mode=mode,
            local_cache_dir=local_cache_dir,
            validation_offset=validation_offset,
        )
        icl_val_dataset = ICLDatasetForForecasting(
            base_dataset=validation_dataset,
            min_examples=min_examples,
            max_examples=max_examples,
            min_past_for_examples=min_past_for_examples,
        )

        return icl_val_dataset
    else:
        raise ValueError(f"Unknown mode: {mode}")


def convert_past_future_datasets_to_icl_input(
    past_data: Dataset, future_data: Dataset,
    target_columns: list[str], 
    past_dynamic_columns: list[str], # time-varying features only available until the forecast start
    known_dynamic_columns: list[str], # time-varying features available for both past and future 
    prediction_length: int,
    context_length: int,
    min_examples: int,
    max_examples: int,
    min_past_for_examples: int,
    use_exogenous: bool = True, # whether to use both past_dynamic_columns and/or known_dynamic_columns
    use_past_dynamic: bool = True, # whether to use past_dynamic_columns (only if use_exogenous is True)
    use_known_dynamic: bool = True, # whether to use known_dynamic_columns (only if use_exogenous is True)
) -> tuple[list[dict], list[str], list[str], list[str]]:
    """
    Convert past and future datasets into an ICLDatasetForForecasting for in-context learning.
    This function processes historical and future data to create a dataset suitable for 
    in-context learning forecasting tasks. It handles different types of covariates and
    creates properly formatted input with target values and exogenous features.
    Args:
        past_data (Dataset): Historical dataset containing past observations and covariates.
        future_data (Dataset): Future dataset containing known future covariates.
        target_columns (list[str]): Names of columns containing target values to forecast.
        past_dynamic_columns (list[str]): Time-varying features only available until forecast start.
        known_dynamic_columns (list[str]): Time-varying features available for both past and future.
        prediction_length (int): Number of time steps to forecast into the future.
        context_length (int): Length of historical context to use for forecasting.
        min_examples (int, optional): Minimum number of examples for ICL. Defaults to 1.
        max_examples (int, optional): Maximum number of examples for ICL. Defaults to 3.
        min_past_for_examples (int, optional): Minimum past length required for examples. Defaults to 4.
        use_exogenous (bool, optional): Whether to include exogenous variables. Defaults to True.
        use_past_dynamic (bool, optional): Whether to include past dynamic features. 
            Only effective if use_exogenous is True. Defaults to True.
        use_known_dynamic (bool, optional): Whether to include known dynamic features.
            Only effective if use_exogenous is True. Defaults to True.
    Returns:
        tuple[list[dict], list[str], list[str], list[str]]: A tuple containing:
            - list[dict]: Pipeline-ready dicts (Format 1) for iAmTimePipeline.predict / predict_quantiles
            - list[str]: Target column names used
            - list[str]: Past dynamic column names used (may be empty based on flags)
            - list[str]: Known dynamic column names used (may be empty based on flags)
    Note:
        - Target values are padded with NaN for future time steps
        - Past dynamic features are padded with NaN for prediction horizon
        - Known dynamic features are concatenated from past and future data
        - If use_exogenous is False, all covariate columns are ignored
        - The function creates a iAmTimeDataset internally before wrapping it in ICLDatasetForForecasting
    """
    if not use_exogenous:
        past_dynamic_columns = []
        known_dynamic_columns = []
    else:
        if not use_past_dynamic:
            past_dynamic_columns = []
        if not use_known_dynamic:
            known_dynamic_columns = []
    num_past_covariates: int = len(past_dynamic_columns)
    num_known_covariates: int = len(known_dynamic_columns)

    target_data = past_data.select_columns(target_columns).with_format("numpy")
    
    data_list = []
    for idx, target_row in enumerate(target_data):
        target_row = cast(dict, target_row)
        target_values = target_row["target"]
        target_padded = np.concatenate([target_values, np.full(prediction_length, np.nan)])
        
        if num_past_covariates + num_known_covariates == 0:
            exogenous = np.empty((0,)) # No other variates - exogenous is empty
        else:
            exogenous_features = []
            exogenous_future_features = []

            # Gather raw past/future covariate arrays for this row
            all_dynamic_columns = past_dynamic_columns + known_dynamic_columns
            if len(all_dynamic_columns) > 0:
                past_cov_data = past_data.select_columns(all_dynamic_columns).with_format("numpy")
                past_row = past_cov_data[idx]
                for col in all_dynamic_columns:
                    exogenous_features.append(past_row[col])
            if len(known_dynamic_columns) > 0:
                future_known_data = future_data.select_columns(known_dynamic_columns).with_format("numpy")
                future_row = future_known_data[idx]
                for col in known_dynamic_columns:
                    exogenous_future_features.append(future_row[col])

            # Stack into 2-d arrays and encode categoricals using shared utility
            exog_past_2d = np.stack(exogenous_features, axis=0) if exogenous_features else np.empty((0, len(target_values)))
            exog_future_2d = np.stack(exogenous_future_features, axis=0) if exogenous_future_features else None
            exog_past_2d, exog_future_2d = encode_categorical_exog(
                target=target_values,
                exog_past=exog_past_2d,
                exog_future=exog_future_2d,
            )

            # Build final exogenous array: past_dynamic padded with NaN, known_dynamic with future values
            final_rows = []
            for i, col in enumerate(past_dynamic_columns):
                padded = np.concatenate([exog_past_2d[i].astype(np.float64), np.full(prediction_length, np.nan)])
                final_rows.append(padded)
            for j, col in enumerate(known_dynamic_columns):
                past_idx = len(past_dynamic_columns) + j
                fut_vals = exog_future_2d[j].astype(np.float64) if exog_future_2d is not None else np.full(prediction_length, np.nan)
                full = np.concatenate([exog_past_2d[past_idx].astype(np.float64), fut_vals])
                final_rows.append(full)

            exogenous = np.stack(final_rows, axis=0) if final_rows else np.empty((0,))
        
        data_list.append({
            'target': target_padded,
            'exogenous': exogenous
        })

    raw_dataset = iAmTimeDataset(
        datasets=[Dataset.from_list(data_list)],
        probabilities=[1.0],
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past_for_examples,
        imputation_method=None,
        mode="validation",
    )
    icl_dataset = ICLDatasetForForecasting(
        base_dataset=raw_dataset,
        min_examples=min_examples,
        max_examples=max_examples,
        min_past_for_examples=min_past_for_examples,
    )

    # Keys accepted by iAmTimePipeline._convert_to_icl_inputs (Format 1)
    _PIPELINE_KEYS = {
        "example_target_histories", "example_target_futures",
        "example_exog_histories", "example_exog_futures",
        "query_target_history", "query_exog_history", "query_exog_future",
        "query_target_future",
    }
    pipeline_inputs = [
        {k: v for k, v in item.items() if k in _PIPELINE_KEYS}
        for item in icl_dataset
    ]

    return pipeline_inputs, target_columns, past_dynamic_columns, known_dynamic_columns
