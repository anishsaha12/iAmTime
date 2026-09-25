import numpy as np
import torch
from sklearn.preprocessing import TargetEncoder


def encode_categorical_exog(
    target: np.ndarray,
    exog_past: np.ndarray,
    exog_future: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode non-numeric (categorical) rows in exogenous arrays using TargetEncoder.

    TargetEncoder replaces each category with the mean of the target for that category.

    Operates on the ICL-level unnamed arrays: a 1-d target and 2-d exog arrays
    where each row is a covariate time series.

    Parameters
    ----------
    target
        1-d numeric array of target values used to fit the encoder.
    exog_past
        2-d array of shape (num_exog, hist_len). Rows may be non-numeric.
    exog_future
        2-d array of shape (num_exog, fut_len) or None. Corresponding rows
        are transformed using encoders fit on ``exog_past``.

    Returns
    -------
    (encoded_past, encoded_future) with categorical rows replaced by
    their target-encoded numeric equivalents. Numeric rows are unchanged.
    """
    exog_past = np.asarray(exog_past)
    if exog_past.ndim < 2 or exog_past.shape[0] == 0:
        return exog_past, exog_future

    target = np.asarray(target, dtype=np.float64)
    has_future = exog_future is not None and np.asarray(exog_future).size > 0
    if has_future:
        exog_future = np.asarray(exog_future)

    any_encoded = False
    encoded_past_rows: list[np.ndarray] = []
    encoded_future_rows: list[np.ndarray] = []

    num_past = exog_past.shape[0]
    num_future = exog_future.shape[0] if has_future else 0

    for i in range(num_past):
        row = np.asarray(exog_past[i])
        if row.size > 0 and not np.issubdtype(row.dtype, np.number):
            encoder = TargetEncoder(target_type="continuous", smooth=1.0)
            X = row.astype(str).reshape(-1, 1)
            y = target[: len(X)]
            mask = np.isfinite(y)
            encoder.fit(X[mask], y[mask])
            encoded_past_rows.append(
                encoder.transform(row.astype(str).reshape(-1, 1)).ravel()
            )
            if has_future and i < num_future:
                fut_row = np.asarray(exog_future[i])
                encoded_future_rows.append(
                    encoder.transform(fut_row.astype(str).reshape(-1, 1)).ravel()
                )
            any_encoded = True
        else:
            encoded_past_rows.append(row)
            if has_future and i < num_future:
                encoded_future_rows.append(np.asarray(exog_future[i]))

    # Pass through any extra future rows that have no corresponding past row.
    # Numeric rows are kept as-is; categorical rows without a past counterpart
    # cannot be target-encoded, so they are replaced with NaN.
    for i in range(num_past, num_future):
        fut_row = np.asarray(exog_future[i])
        if fut_row.size > 0 and not np.issubdtype(fut_row.dtype, np.number):
            encoded_future_rows.append(np.full(fut_row.shape, np.nan))
            any_encoded = True
        else:
            encoded_future_rows.append(fut_row)

    if not any_encoded:
        return exog_past, exog_future

    encoded_past = np.stack([r.astype(np.float64) for r in encoded_past_rows])
    encoded_future = (
        np.stack([r.astype(np.float64) for r in encoded_future_rows])
        if has_future
        else exog_future
    )
    return encoded_past, encoded_future


def encode_categorical_exog_in_icl_dict(icl_input: dict) -> dict:
    """Encode categorical exogenous rows in a single ICL-format dictionary.

    Processes both query and per-example covariates in-place, using the
    corresponding target history as the regression signal for TargetEncoder.

    Parameters
    ----------
    icl_input
        A dictionary in ICL format with keys such as
        ``query_target_history``, ``query_exog_history``,
        ``query_exog_future``, ``example_target_histories``,
        ``example_exog_histories``, ``example_exog_futures``, etc.

    Returns
    -------
    The same dictionary with categorical exogenous rows replaced by their
    target-encoded numeric equivalents.
    """
    # --- query ---
    query_target = np.asarray(icl_input.get("query_target_history", []))
    query_exog_hist = icl_input.get("query_exog_history")
    query_exog_fut = icl_input.get("query_exog_future")
    if query_target.size > 0 and query_exog_hist is not None and np.asarray(query_exog_hist).size > 0:
        enc_hist, enc_fut = encode_categorical_exog(
            target=query_target,
            exog_past=query_exog_hist,
            exog_future=query_exog_fut,
        )
        icl_input["query_exog_history"] = enc_hist
        if enc_fut is not None:
            icl_input["query_exog_future"] = enc_fut

    # --- examples ---
    ex_targets = icl_input.get("example_target_histories", [])
    ex_exog_hists = icl_input.get("example_exog_histories", [])
    ex_exog_futs = icl_input.get("example_exog_futures", [])
    if isinstance(ex_targets, (list, np.ndarray)) and len(ex_targets) > 0:
        for j in range(len(ex_targets)):
            ex_target_j = np.asarray(ex_targets[j])
            ex_exog_hist_j = np.asarray(ex_exog_hists[j]) if j < len(ex_exog_hists) else None
            if ex_target_j.size > 0 and ex_exog_hist_j is not None and ex_exog_hist_j.size > 0:
                ex_exog_fut_j = np.asarray(ex_exog_futs[j]) if j < len(ex_exog_futs) else None
                enc_h, enc_f = encode_categorical_exog(
                    target=ex_target_j,
                    exog_past=ex_exog_hist_j,
                    exog_future=ex_exog_fut_j,
                )
                ex_exog_hists[j] = enc_h
                if enc_f is not None and j < len(ex_exog_futs):
                    ex_exog_futs[j] = enc_f

    return icl_input


class GetPaddedBatchInputsForICL:
    def __init__(self, model_icl_inputs: list[dict[str, torch.Tensor]]):
        # Encode any categorical exogenous covariates before padding
        for icl_input in model_icl_inputs:
            encode_categorical_exog_in_icl_dict(icl_input)

        self._model_icl_inputs = model_icl_inputs
        self._num_inputs = len(self._model_icl_inputs)
        self._batch_measurements = None

        for input_data in self._model_icl_inputs:
            assert "example_target_histories" in input_data
            assert "example_target_futures" in input_data
            assert "example_exog_histories" in input_data
            assert "example_exog_futures" in input_data
            assert "query_target_history" in input_data
            assert "query_exog_history" in input_data
            assert "query_exog_future" in input_data
            assert "query_target_future" in input_data
            for key in input_data:
                val = input_data[key]
                if isinstance(val, list):
                    for item in val:
                        assert isinstance(item, (np.ndarray, torch.Tensor)), (
                            f"Expected np.ndarray or torch.Tensor for items in '{key}', got {type(item)}"
                        )
                else:
                    assert isinstance(val, (np.ndarray, torch.Tensor)), (
                        f"Expected np.ndarray or torch.Tensor for '{key}', got {type(val)}"
                    )

    def _get_input_measurements(
        self,
        example_target_histories: torch.Tensor = [],
        example_target_futures: torch.Tensor = [],
        example_exog_histories: torch.Tensor = [],
        example_exog_futures: torch.Tensor = [],
        query_target_history: torch.Tensor = [],
        query_exog_history: torch.Tensor = [],
        query_exog_future: torch.Tensor = [],
        query_target_future: torch.Tensor = [],
    ) -> dict[str, int]:
        # Padding
        hist_max_target_len = np.max([ex_tr_h.shape for ex_tr_h in example_target_histories]) if len(example_target_histories) > 0 else 0
        futu_max_target_len = np.max([ex_tr_f.shape for ex_tr_f in example_target_futures]) if len(example_target_futures) > 0 else 0
        hist_max_num_exog, hist_max_exog_len = np.max([ex_eg_h.shape for ex_eg_h in example_exog_histories], axis=0) if len(example_exog_histories) > 0 else (0, 0)
        futu_max_num_exog, futu_max_exog_len = np.max([ex_eg_f.shape for ex_eg_f in example_exog_futures], axis=0) if len(example_exog_futures) > 0 else (0, 0)
        num_examples_in_input = np.max((
            len(example_target_histories), len(example_target_futures),
            len(example_exog_histories), len(example_exog_futures)
        ))
        hist_max_len_in_input_examples = np.max((hist_max_target_len, hist_max_exog_len))
        futu_max_len_in_input_examples = np.max((futu_max_target_len, futu_max_exog_len))
        num_exog_in_input_examples = np.max((hist_max_num_exog, futu_max_num_exog))

        hist_max_len_in_input_query = np.max(
            [ex_qu_h.shape[0] for ex_qu_h in query_exog_history]+
            [query_target_history.shape[0]], 
            axis=0
        )
        futu_max_len_in_input_query = np.max([ex_qu_f.shape[0] for ex_qu_f in query_exog_future]) if len(query_exog_future) > 0 else 0
        futu_max_len_in_input_query = np.max((futu_max_len_in_input_query, query_target_future.shape[0]))
        num_exog_in_input_query = np.max([query_exog_history.shape[0], query_exog_future.shape[0]]) if (
            (len(query_exog_history) > 0) and (len(query_exog_future) > 0)
        ) else query_exog_history.shape[0] if (
            (len(query_exog_history) > 0) and (len(query_exog_future) == 0)
        ) else query_exog_future.shape[0] if (
            (len(query_exog_history) == 0) and (len(query_exog_future) > 0)
        ) else 0

        return {
            "num_examples_in_input": num_examples_in_input,
            "hist_max_len_in_input_examples": hist_max_len_in_input_examples,
            "futu_max_len_in_input_examples": futu_max_len_in_input_examples,
            "num_exog_in_input_examples": num_exog_in_input_examples,
            "hist_max_len_in_input_query": hist_max_len_in_input_query,
            "futu_max_len_in_input_query": futu_max_len_in_input_query,
            "num_exog_in_input_query": num_exog_in_input_query
        }

    def _get_batch_measurements(self) -> dict[str, int]:
        """
        Call self._get_input_measurements for each input in self._model_icl_inputs
        Return the maximum of each attribute across the inputs.
        """
        batch_measurements = {}
        for input_data in self._model_icl_inputs:
            measurements = self._get_input_measurements(
                example_target_histories=input_data["example_target_histories"],
                example_target_futures=input_data["example_target_futures"],
                example_exog_histories=input_data["example_exog_histories"],
                example_exog_futures=input_data["example_exog_futures"],
                query_target_history=input_data["query_target_history"],
                query_exog_history=input_data["query_exog_history"],
                query_exog_future=input_data["query_exog_future"],
                query_target_future=input_data["query_target_future"]
            )
            for key, value in measurements.items():
                if key not in batch_measurements:
                    batch_measurements[key] = value
                else:
                    batch_measurements[key] = np.max((batch_measurements[key], value))
        hist_max_len_in_input = np.max([
            batch_measurements['hist_max_len_in_input_examples'], 
            batch_measurements['hist_max_len_in_input_query']
        ])
        futu_max_len_in_input = np.max([
            batch_measurements['futu_max_len_in_input_examples'], 
            batch_measurements['futu_max_len_in_input_query']
        ])
        batch_measurements['hist_max_len_in_input_examples'] = hist_max_len_in_input
        batch_measurements['hist_max_len_in_input_query'] = hist_max_len_in_input
        batch_measurements['futu_max_len_in_input_examples'] = futu_max_len_in_input
        batch_measurements['futu_max_len_in_input_query'] = futu_max_len_in_input
        return batch_measurements

    def _pad_input(
            self, 
            example_target_histories: torch.Tensor = None,
            example_target_futures: torch.Tensor = None,
            example_exog_histories: torch.Tensor = None,
            example_exog_futures: torch.Tensor = None,
            query_target_history: torch.Tensor = None,
            query_exog_history: torch.Tensor = None,
            query_exog_future: torch.Tensor = None,
            query_target_future: torch.Tensor = None,
            num_examples_in_input: int = None,
            hist_max_len_in_input_examples: int = None,
            futu_max_len_in_input_examples: int = None,
            num_exog_in_input_examples: int = None,
            hist_max_len_in_input_query: int = None,
            futu_max_len_in_input_query: int = None,
            num_exog_in_input_query: int = None,
            pad_using_value = np.nan,
    ) -> dict[str, torch.Tensor]:
        """
        Pad the input data tensors to the maximum lengths specified in batch_measurements.
        """
        # Padding Examples
        padded_example_target_histories = []
        padded_example_exog_histories = []

        padded_example_target_futures = []
        padded_example_exog_futures = []

        for examp in range(num_examples_in_input):
            if examp < len(example_target_histories):
                curr_example_shape = example_target_histories[examp].shape
                if curr_example_shape[0] < hist_max_len_in_input_examples:
                    padded_example_target_histories.append(np.concatenate([
                        np.full(hist_max_len_in_input_examples-curr_example_shape[0], pad_using_value), 
                        example_target_histories[examp]
                    ]))
                else:
                    padded_example_target_histories.append(example_target_histories[examp])
            else:
                padded_example_target_histories.append(np.full(hist_max_len_in_input_examples, pad_using_value))

        for examp in range(num_examples_in_input):
            if examp < len(example_target_futures):
                curr_example_shape = example_target_futures[examp].shape
                if curr_example_shape[0] < futu_max_len_in_input_examples:
                    padded_example_target_futures.append(np.concatenate([
                        example_target_futures[examp],
                        np.full(futu_max_len_in_input_examples-curr_example_shape[0], pad_using_value)
                    ]))
                else:
                    padded_example_target_futures.append(example_target_futures[examp])
            else:
                padded_example_target_futures.append(np.full(futu_max_len_in_input_examples, pad_using_value))

        for examp in range(num_examples_in_input):
            if examp < len(example_exog_histories):
                curr_example_shape = example_exog_histories[examp].shape
                curr_example_hist_padded = []
                for exog in range(num_exog_in_input_examples):
                    if (exog+1) <= curr_example_shape[0]:
                        if curr_example_shape[1] < hist_max_len_in_input_examples:
                            curr_example_hist_padded.append(np.concatenate([
                                np.full(hist_max_len_in_input_examples-curr_example_shape[1], pad_using_value), 
                                example_exog_histories[examp][exog]
                            ]))
                        else:
                            curr_example_hist_padded.append(example_exog_histories[examp][exog])
                    else:
                        curr_example_hist_padded.append(np.full(hist_max_len_in_input_examples, pad_using_value))
                padded_example_exog_histories.append(
                    np.stack(curr_example_hist_padded, axis=0) if (
                        len(curr_example_hist_padded) > 0
                    ) else np.full((num_exog_in_input_examples, hist_max_len_in_input_examples), pad_using_value)
                )
            else:
                padded_example_exog_histories.append(
                    np.full((num_exog_in_input_examples, hist_max_len_in_input_examples), pad_using_value)
                )

        for examp in range(num_examples_in_input):
            if examp < len(example_exog_futures):
                curr_example_shape = example_exog_futures[examp].shape
                curr_example_fut_padded = []
                for exog in range(num_exog_in_input_examples):
                    if (exog+1) <= curr_example_shape[0]:
                        if curr_example_shape[1] < futu_max_len_in_input_examples:
                            curr_example_fut_padded.append(np.concatenate([
                                example_exog_futures[examp][exog],
                                np.full(futu_max_len_in_input_examples-curr_example_shape[1], pad_using_value)
                            ]))
                        else:
                            curr_example_fut_padded.append(example_exog_futures[examp][exog])
                    else:
                        curr_example_fut_padded.append(np.full(futu_max_len_in_input_examples, pad_using_value))
                padded_example_exog_futures.append(
                    np.stack(curr_example_fut_padded, axis=0) if (
                        len(curr_example_fut_padded) > 0
                    ) else np.full((num_exog_in_input_examples, futu_max_len_in_input_examples), pad_using_value)
                )
            else:
                padded_example_exog_futures.append(
                    np.full((num_exog_in_input_examples, futu_max_len_in_input_examples), pad_using_value)
                )

        # Convert to numpy arrays
        example_target_histories = np.stack(padded_example_target_histories, axis=0) if (
            len(padded_example_target_histories) > 0
        ) else np.stack([
            np.full(
                (1 if hist_max_len_in_input_examples == 0 else hist_max_len_in_input_examples), 
                pad_using_value
            )
        ], axis=0)
        example_target_futures = np.stack(padded_example_target_futures, axis=0) if (
            len(padded_example_target_futures) > 0
        ) else np.stack([
            np.full(
                (1 if futu_max_len_in_input_examples == 0 else futu_max_len_in_input_examples), 
                pad_using_value
            )
        ], axis=0)
        example_exog_histories = np.stack(padded_example_exog_histories, axis=0) if (
            len(padded_example_exog_histories) > 0
        ) else np.stack([
            np.full((
                1, 
                (1 if hist_max_len_in_input_examples == 0 else hist_max_len_in_input_examples)
            ), pad_using_value)
        ], axis=0)
        example_exog_futures = np.stack(padded_example_exog_futures, axis=0) if (
            len(padded_example_exog_futures) > 0
        ) else np.stack([
            np.full((
                1, 
                (1 if futu_max_len_in_input_examples == 0 else futu_max_len_in_input_examples)
            ), pad_using_value)
        ], axis=0)

        # Padding Query
        padded_query_exog_future = []

        query_target_history = np.concatenate([
            np.full(hist_max_len_in_input_query-query_target_history.shape[0], pad_using_value), 
            query_target_history
        ])

        query_exog_hist_shape = query_exog_history.shape if len(query_exog_history) > 0 else (0, 0)
        padded_query_exog_history = []
        for exog in range(num_exog_in_input_query):
            if (exog+1) <= query_exog_hist_shape[0]:
                if query_exog_hist_shape[1] < hist_max_len_in_input_query:
                    padded_query_exog_history.append(np.concatenate([
                        np.full(hist_max_len_in_input_query-query_exog_hist_shape[1], pad_using_value), 
                        query_exog_history[exog]
                    ]))
                else:
                    padded_query_exog_history.append(query_exog_history[exog])
            else:
                padded_query_exog_history.append(np.full(hist_max_len_in_input_query, pad_using_value))
        query_exog_history = np.stack(padded_query_exog_history, axis=0) if (
            len(padded_query_exog_history) > 0
        ) else np.stack([
            np.full(
                (1 if hist_max_len_in_input_query == 0 else hist_max_len_in_input_query), 
                pad_using_value
            )
        ], axis=0)

        query_exog_futu_shape = query_exog_future.shape if len(query_exog_future) > 0 else (0, 0)
        padded_query_exog_future = []
        for exog in range(num_exog_in_input_query):
            if (exog+1) <= query_exog_futu_shape[0]:
                if query_exog_futu_shape[1] < futu_max_len_in_input_query:
                    padded_query_exog_future.append(np.concatenate([
                        query_exog_future[exog],
                        np.full(futu_max_len_in_input_query-query_exog_futu_shape[1], pad_using_value)
                    ]))
                else:
                    padded_query_exog_future.append(query_exog_future[exog])
            else:
                padded_query_exog_future.append(np.full(futu_max_len_in_input_query, pad_using_value))
        query_exog_future = np.stack(padded_query_exog_future, axis=0) if (
            len(padded_query_exog_future) > 0
        ) else np.stack([
            np.full(
                (1 if futu_max_len_in_input_query == 0 else futu_max_len_in_input_query), 
                pad_using_value
            )
        ], axis=0)

        query_target_future = np.concatenate([
            query_target_future,
            np.full(futu_max_len_in_input_query-query_target_future.shape[0], pad_using_value)
        ])

        return {
            "example_target_histories": example_target_histories,
            "example_target_futures": example_target_futures,
            "example_exog_histories": example_exog_histories,
            "example_exog_futures": example_exog_futures,
            "query_target_history": query_target_history,
            "query_exog_history": query_exog_history,
            "query_exog_future": query_exog_future,
            "query_target_future": query_target_future,
        }
    
    def get_padded_batch(self) -> dict[str, torch.Tensor]:
        """
        Call self._pad_input for each input in self._model_icl_inputs,
         to the maximum length within the batch obtained from self._get_batch_measurements
        Return a Batch of inputs
        """
        self._batch_measurements = self._get_batch_measurements()
        padded_inputs = []
        for input_data in self._model_icl_inputs:
            padded_input = self._pad_input(
                example_target_histories=input_data["example_target_histories"],
                example_target_futures=input_data["example_target_futures"],
                example_exog_histories=input_data["example_exog_histories"],
                example_exog_futures=input_data["example_exog_futures"],
                query_target_history=input_data["query_target_history"],
                query_exog_history=input_data["query_exog_history"],
                query_exog_future=input_data["query_exog_future"],
                query_target_future=input_data["query_target_future"],
                **self._batch_measurements
            )
            padded_inputs.append(padded_input)

        # Create to torch batch input by adding batch dimension
        model_icl_inputs_batch = { 
            "example_target_histories": torch.cat([
                torch.tensor(input_i["example_target_histories"], dtype=torch.float32).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "example_target_futures": torch.cat([
                torch.tensor(input_i["example_target_futures"], dtype=torch.float32).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "example_exog_histories": torch.cat([
                torch.tensor(input_i["example_exog_histories"], dtype=torch.float32).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "example_exog_futures": torch.cat([
                torch.tensor(input_i["example_exog_futures"], dtype=torch.float32).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "query_target_history": torch.cat([
                torch.tensor(input_i["query_target_history"], dtype=torch.float32).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "query_exog_history": torch.cat([
                torch.tensor(input_i["query_exog_history"], dtype=torch.float32).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "query_exog_future": torch.cat([
                torch.tensor(input_i["query_exog_future"], dtype=torch.float32).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "query_target_future": torch.cat([
                torch.tensor(input_i["query_target_future"], dtype=torch.float32).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
        }

        return model_icl_inputs_batch

    def get_padded_batch_masks_from_measurements(self, model_icl_input_masks) -> dict[str, torch.Tensor]:
        """
        Call self._pad_input for each input in model_icl_input_masks,
         to the maximum length within the batch using the batch_measurements
         which can be obtained from calling self._get_batch_measurements on the inputs
        Return a Batch of input masks
        """
        assert self._batch_measurements is not None, "Batch measurements are not set"
        padded_inputs = []
        for input_data in model_icl_input_masks:
            padded_input = self._pad_input(
                example_target_histories=input_data["example_target_hist_mask"],
                example_target_futures=input_data["example_target_fut_mask"],
                example_exog_histories=input_data["example_exog_hist_mask"],
                example_exog_futures=input_data["example_exog_fut_mask"],
                query_target_history=input_data["query_target_hist_mask"],
                query_exog_history=input_data["query_exog_hist_mask"],
                query_exog_future=input_data["query_exog_fut_mask"],
                query_target_future=input_data["query_target_fut_mask"],
                **self._batch_measurements,
                pad_using_value=False,
            )
            padded_inputs.append(padded_input)

        # Create to torch batch input by adding batch dimension
        model_icl_input_masks_batch = { 
            "example_target_hist_mask": torch.cat([
                torch.tensor(input_i["example_target_histories"], dtype=torch.bool).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "example_target_fut_mask": torch.cat([
                torch.tensor(input_i["example_target_futures"], dtype=torch.bool).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "example_exog_hist_mask": torch.cat([
                torch.tensor(input_i["example_exog_histories"], dtype=torch.bool).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "example_exog_fut_mask": torch.cat([
                torch.tensor(input_i["example_exog_futures"], dtype=torch.bool).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "query_target_hist_mask": torch.cat([
                torch.tensor(input_i["query_target_history"], dtype=torch.bool).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "query_exog_hist_mask": torch.cat([
                torch.tensor(input_i["query_exog_history"], dtype=torch.bool).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "query_exog_fut_mask": torch.cat([
                torch.tensor(input_i["query_exog_future"], dtype=torch.bool).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
            "query_target_fut_mask": torch.cat([
                torch.tensor(input_i["query_target_future"], dtype=torch.bool).unsqueeze(0)
                for input_i in padded_inputs
            ], dim=0),
        }

        return model_icl_input_masks_batch
    
    def _right_pad_tensor(
            self,
            input_tensors: torch.Tensor,
            total_length: int,
    ) -> torch.Tensor:
        assert isinstance(input_tensors, torch.Tensor)
        curr_length = input_tensors.shape[-1]
        assert curr_length <= total_length
        padding_shape = input_tensors.shape[:-1] + (total_length - curr_length,)
        padding = torch.full(
            size=padding_shape, fill_value=torch.nan, device=input_tensors.device
        )
        padded = torch.concat((input_tensors, padding), dim=-1)
        return padded

    def right_pad_icl_input_tensors(
            self,
            batched_icl_input_tensors: torch.Tensor,
            apply_padding_to_input_attributes: list[str],
            total_length: int,
    ) -> torch.Tensor:
        for input_attribute in apply_padding_to_input_attributes:
            if input_attribute in batched_icl_input_tensors:
                batched_icl_input_tensors[input_attribute] = self._right_pad_tensor(
                    batched_icl_input_tensors[input_attribute], total_length
                )
        return batched_icl_input_tensors
