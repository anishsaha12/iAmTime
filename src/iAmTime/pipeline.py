import logging
from typing import Dict, List, Optional, Tuple
import warnings

from einops import rearrange
import numpy as np
import torch
from transformers import AutoConfig

from .model import iAmTimeModel
from .utils_icl import GetPaddedBatchInputsForICL

logger = logging.getLogger(__file__)

class PipelineRegistry(type):
    REGISTRY: Dict[str, "PipelineRegistry"] = {}

    def __new__(cls, name, bases, attrs):
        """See, https://github.com/faif/python-patterns."""
        new_cls = type.__new__(cls, name, bases, attrs)
        if name is not None:
            cls.REGISTRY[name] = new_cls

        return new_cls


class iAmTimePipeline(metaclass=PipelineRegistry):
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}

    def __init__(self, model: iAmTimeModel):
        self.model = model

    @property
    def icl_input_attributes(self) -> List[str]:
        return [
            "example_target_histories", "example_target_futures", 
            "example_exog_histories", "example_exog_futures",
            "query_target_history", "query_exog_history", "query_exog_future"
        ]

    @property
    def icl_optional_input_attributes(self) -> List[str]:
        return ["query_target_future"]

    @property
    def quantiles(self) -> List[float]:
        return self.model.iamtime_config.quantiles
    
    def _convert_to_icl_inputs(self, icl_inputs) -> dict[str, torch.Tensor]:
        # Implement the processing logic here
        if (
            isinstance(icl_inputs, list) and 
            all([isinstance(x, dict) for x in icl_inputs]) and
            all([
                set(x.keys()).issubset(set(self.icl_input_attributes + self.icl_optional_input_attributes)) 
                for x in icl_inputs
            ])
        ):
            for icl_input in icl_inputs:
                for attribute in self.icl_input_attributes + self.icl_optional_input_attributes:
                    if attribute not in icl_input:
                        icl_input[attribute] = np.array([])
            formatted_icl_inputs = icl_inputs
        elif (
            isinstance(icl_inputs, list) and 
            all([isinstance(x, dict) for x in icl_inputs]) and
            all([
                set(x.keys()).issubset(set(["target", "past_covariates", "future_covariates"])) 
                for x in icl_inputs
            ])
        ):
            formatted_icl_inputs = []
            for icl_input in icl_inputs:
                query_exog_history = []
                query_exog_future = []
                past_keys = list(icl_input["past_covariates"].keys()) if "past_covariates" in icl_input else []
                future_keys = list(icl_input["future_covariates"].keys()) if "future_covariates" in icl_input else []
                all_keys = list(dict.fromkeys(past_keys + future_keys))  # union, preserving order
                for variate_key in all_keys:
                    if variate_key in icl_input.get("past_covariates", {}):
                        query_exog_history.append(np.array(icl_input["past_covariates"][variate_key]))
                    else:
                        query_exog_history.append(None)  # placeholder for missing
                    if variate_key in icl_input.get("future_covariates", {}):
                        query_exog_future.append(np.array(icl_input["future_covariates"][variate_key]))
                    else:
                        query_exog_future.append(None)  # placeholder for missing

                # Assert uniform lengths across provided (non-None) covariates
                hist_lengths = [len(h) for h in query_exog_history if h is not None]
                fut_lengths = [len(f) for f in query_exog_future if f is not None]
                assert len(set(hist_lengths)) <= 1, (
                    f"All past_covariates must have the same length, got {hist_lengths}"
                )
                assert len(set(fut_lengths)) <= 1, (
                    f"All future_covariates must have the same length, got {fut_lengths}"
                )
                hist_len = hist_lengths[0] if hist_lengths else 0
                fut_len = fut_lengths[0] if fut_lengths else 0

                # Left-pad missing histories, right-pad missing futures with NaN
                for i, h in enumerate(query_exog_history):
                    if h is None:
                        query_exog_history[i] = np.full(hist_len, np.nan) if hist_len > 0 else np.array([])
                for i, f in enumerate(query_exog_future):
                    if f is None:
                        query_exog_future[i] = np.full(fut_len, np.nan) if fut_len > 0 else np.array([])

                formatted_icl_inputs.append({
                    "example_target_histories": np.array([]),
                    "example_target_futures": np.array([]),
                    "example_exog_histories": np.array([]),
                    "example_exog_futures": np.array([]),
                    "query_target_history": np.array(icl_input["target"]),
                    "query_exog_history": np.array(query_exog_history),
                    "query_exog_future": np.array(query_exog_future),
                    "query_target_future": np.array([]),
                })
        elif isinstance(icl_inputs, (torch.Tensor, np.ndarray)):
            assert icl_inputs.ndim == 2, "Expected input shape (batch_size, sequence_length)"
            formatted_icl_inputs = []
            for icl_input in icl_inputs:
                formatted_icl_inputs.append({
                    "example_target_histories": np.array([]),
                    "example_target_futures": np.array([]),
                    "example_exog_histories": np.array([]),
                    "example_exog_futures": np.array([]),
                    "query_target_history": np.array(icl_input),
                    "query_exog_history": np.array([]),
                    "query_exog_future": np.array([]),
                    "query_target_future": np.array([]),
                })
        elif isinstance(icl_inputs, list) and all([
            isinstance(x, (torch.Tensor, np.ndarray)) for x in icl_inputs
        ]) and all([
            x.ndim == 1 for x in icl_inputs
        ]):
            formatted_icl_inputs = []
            for icl_input in icl_inputs:
                formatted_icl_inputs.append({
                    "example_target_histories": np.array([]),
                    "example_target_futures": np.array([]),
                    "example_exog_histories": np.array([]),
                    "example_exog_futures": np.array([]),
                    "query_target_history": np.array(icl_input),
                    "query_exog_history": np.array([]),
                    "query_exog_future": np.array([]),
                    "query_target_future": np.array([]),
                })
        else:
            raise ValueError("Unexpected inputs format")
        return formatted_icl_inputs

    def _expand_multivariate(self, icl_inputs):
        """Expand multivariate targets into multiple univariate inputs.

        When a target has shape ``(n_variates, length)`` with ``n_variates > 1``,
        creates ``n_variates`` copies.  In each copy one variate is the 1-D
        target and the remaining variates become additional exogenous covariates.

        Returns
        -------
        expanded_inputs
            Inputs with all targets guaranteed 1-D (same or compatible format).
        group_sizes
            List of ints – how many expanded items each original batch element
            produced.  ``sum(group_sizes) == len(expanded_inputs)``.
        """
        group_sizes: List[int] = []
        expanded: list = []

        # ── List of dicts (Format 1 or 2) ────────────────────────────────
        if isinstance(icl_inputs, list) and all(isinstance(x, dict) for x in icl_inputs):
            icl_keys = set(
                self.icl_input_attributes + self.icl_optional_input_attributes
            )
            is_icl = all(set(x.keys()).issubset(icl_keys) for x in icl_inputs)
            is_tgt = (not is_icl) and all(
                set(x.keys()).issubset({"target", "past_covariates", "future_covariates"})
                for x in icl_inputs
            )

            if is_icl:
                # Format 1: ICL-attribute dicts
                any_mv = any(
                    np.asarray(x.get("query_target_history", [])).ndim == 2
                    and np.asarray(x.get("query_target_history", [])).shape[0] > 1
                    for x in icl_inputs
                )
                if not any_mv:
                    return icl_inputs, [1] * len(icl_inputs)

                for item in icl_inputs:
                    qth = np.asarray(item.get("query_target_history", []))
                    if qth.ndim == 2 and qth.shape[0] > 1:
                        n_v = qth.shape[0]
                        group_sizes.append(n_v)
                        qtf = np.asarray(item.get("query_target_future", []))
                        mv_fut = qtf.ndim == 2 and qtf.shape[0] == n_v
                        ex_hist = np.asarray(item.get("query_exog_history", []))
                        ex_fut = np.asarray(item.get("query_exog_future", []))
                        for v in range(n_v):
                            ni = dict(item)  # shallow copy
                            ni["query_target_history"] = qth[v]
                            other_h = np.delete(qth, v, axis=0)
                            ni["query_exog_history"] = (
                                np.concatenate([other_h, ex_hist], axis=0)
                                if ex_hist.ndim == 2 and ex_hist.shape[0] > 0
                                else other_h
                            )
                            if mv_fut:
                                ni["query_target_future"] = qtf[v]
                                other_f = np.delete(qtf, v, axis=0)
                                ni["query_exog_future"] = (
                                    np.concatenate([other_f, ex_fut], axis=0)
                                    if ex_fut.ndim == 2 and ex_fut.shape[0] > 0
                                    else other_f
                                )
                            else:
                                # Prepend NaN placeholders for the other target
                                # variates so exog indices stay aligned between
                                # history and future.
                                n_other = n_v - 1
                                if ex_fut.ndim == 2 and ex_fut.shape[0] > 0:
                                    fut_len = ex_fut.shape[1]
                                    nan_futs = np.full((n_other, fut_len), np.nan)
                                    ni["query_exog_future"] = np.concatenate(
                                        [nan_futs, ex_fut], axis=0
                                    )
                                else:
                                    ni["query_exog_future"] = np.empty((n_other, 0))
                            expanded.append(ni)
                    else:
                        group_sizes.append(1)
                        expanded.append(item)
                return expanded, group_sizes

            if is_tgt:
                # Format 2: target / past_covariates / future_covariates dicts
                any_mv = any(
                    np.asarray(x.get("target", [])).ndim == 2
                    and np.asarray(x.get("target", [])).shape[0] > 1
                    for x in icl_inputs
                )
                if not any_mv:
                    return icl_inputs, [1] * len(icl_inputs)

                for item in icl_inputs:
                    tgt = np.asarray(item["target"])
                    if tgt.ndim == 2 and tgt.shape[0] > 1:
                        n_v = tgt.shape[0]
                        group_sizes.append(n_v)
                        for v in range(n_v):
                            ni = {"target": tgt[v]}
                            past = dict(item.get("past_covariates", {}))
                            other = np.delete(tgt, v, axis=0)
                            for k in range(other.shape[0]):
                                past[f"_target_variate_{k}"] = other[k]
                            if past:
                                ni["past_covariates"] = past
                            if "future_covariates" in item:
                                ni["future_covariates"] = dict(item["future_covariates"])
                            expanded.append(ni)
                    else:
                        group_sizes.append(1)
                        expanded.append(item)
                return expanded, group_sizes

            # dict format not recognised – pass through
            return icl_inputs, [1] * len(icl_inputs)

        # ── 2-D / 3-D tensor or array (Format 3) ────────────────────────
        if isinstance(icl_inputs, (torch.Tensor, np.ndarray)):
            arr = np.asarray(icl_inputs)
            if arr.ndim == 3:
                B, n_v, _L = arr.shape
                for b in range(B):
                    group_sizes.append(n_v)
                    for v in range(n_v):
                        expanded.append({
                            "query_target_history": arr[b, v],
                            "query_exog_history": np.delete(arr[b], v, axis=0),
                            "query_exog_future": np.array([]),
                            "example_target_histories": np.array([]),
                            "example_target_futures": np.array([]),
                            "example_exog_histories": np.array([]),
                            "example_exog_futures": np.array([]),
                        })
                return expanded, group_sizes
            return icl_inputs, [1] * arr.shape[0]

        # ── List of 1-D / 2-D arrays (Format 4) ─────────────────────────
        if isinstance(icl_inputs, list) and all(
            isinstance(x, (torch.Tensor, np.ndarray)) for x in icl_inputs
        ):
            if not any(
                np.asarray(x).ndim == 2 and np.asarray(x).shape[0] > 1
                for x in icl_inputs
            ):
                return icl_inputs, [1] * len(icl_inputs)

            for x in icl_inputs:
                a = np.asarray(x)
                if a.ndim == 2 and a.shape[0] > 1:
                    n_v = a.shape[0]
                    group_sizes.append(n_v)
                    for v in range(n_v):
                        expanded.append({
                            "query_target_history": a[v],
                            "query_exog_history": np.delete(a, v, axis=0),
                            "query_exog_future": np.array([]),
                            "example_target_histories": np.array([]),
                            "example_target_futures": np.array([]),
                            "example_exog_histories": np.array([]),
                            "example_exog_futures": np.array([]),
                        })
                else:
                    group_sizes.append(1)
                    expanded.append({
                        "query_target_history": a.ravel(),
                        "query_exog_history": np.array([]),
                        "query_exog_future": np.array([]),
                        "example_target_histories": np.array([]),
                        "example_target_futures": np.array([]),
                        "example_exog_histories": np.array([]),
                        "example_exog_futures": np.array([]),
                    })
            return expanded, group_sizes

        # Fallback – unknown format, let _convert_to_icl_inputs handle it
        n = len(icl_inputs) if isinstance(icl_inputs, (list, tuple)) else 1
        return icl_inputs, [1] * n

    @staticmethod
    def _regroup_predictions(predictions, group_sizes):
        """Regroup flat predictions back into per-input variate groups.

        Parameters
        ----------
        predictions : torch.Tensor
            Shape ``(exploded_batch_size, ...)``.
        group_sizes : list[int]
            One entry per original batch element.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            If all *group_sizes* are equal: ``(batch_size, n_variates, ...)``.
            Otherwise a list of tensors, each ``(n_variates_i, ...)``.
        """
        groups = []
        offset = 0
        for g in group_sizes:
            groups.append(predictions[offset : offset + g])
            offset += g
        if len(set(group_sizes)) == 1:
            return torch.stack(groups)
        return groups

    def _process_icl_inputs(self, icl_inputs, prediction_length) -> dict[str, torch.Tensor]:
        """
        Get forecasts for the given ICL input containing examples and query time series.

        Parameters
        ---------------------
        icl_inputs
            A list of dictionaries containing the ICL input tensors.
        prediction_length
            The length of the prediction horizon.
        """
        model_context_length = self.model.iamtime_config.context_length

        icl_inputs = self._convert_to_icl_inputs(icl_inputs)

        # Implement the processing logic here
        batch_input_for_icl = GetPaddedBatchInputsForICL(
            model_icl_inputs=icl_inputs
        )
        icl_input_tensors_batch = batch_input_for_icl.get_padded_batch()
        icl_input_tensors_batch = batch_input_for_icl.right_pad_icl_input_tensors(
            batched_icl_input_tensors=icl_input_tensors_batch,
            apply_padding_to_input_attributes=[
                "example_target_futures", "example_exog_futures", "query_exog_future"
            ],
            total_length=prediction_length
        )

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        batch_icl_inputs = {}
        for icl_input_attribute in self.icl_input_attributes + self.icl_optional_input_attributes:
            if (
                (icl_input_attribute in self.icl_optional_input_attributes) and 
                (icl_input_attribute not in icl_input_tensors_batch)
            ):
                continue
            context_tensor = icl_input_tensors_batch[icl_input_attribute]
            if context_tensor.shape[-1] > model_context_length:
                context_tensor = context_tensor[..., -model_context_length:]
            context_tensor = context_tensor.to(
                device=self.model.device,
                dtype=torch.float32,
            )
            batch_icl_inputs[icl_input_attribute] = context_tensor
        return batch_icl_inputs

    def _resolve_prediction_length(
        self, icl_inputs, prediction_length: Optional[int]
    ) -> int:
        """Infer prediction_length from inputs or fall back to the model default.

        Priority:
        1. ``query_exog_future`` last-axis length (if present and non-empty).
        2. ``query_target_future`` last-axis length (if present and non-empty).
        3. Explicit ``prediction_length`` argument.
        4. ``model_prediction_length`` (patch_size * max_output_steps).

        If a future component is found *and* ``prediction_length`` was also
        supplied, they must agree or a ``ValueError`` is raised.
        """
        inferred = None
        source = None

        # Scan inputs for future-component lengths
        items = icl_inputs if isinstance(icl_inputs, list) and all(
            isinstance(x, dict) for x in icl_inputs
        ) else []
        for item in items:
            # Format 1 keys
            for key in ("query_exog_future", "query_target_future"):
                arr = item.get(key)
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.size == 0:
                    continue
                length = arr.shape[-1]
                if inferred is None:
                    inferred = length
                    source = key
                elif inferred != length:
                    raise ValueError(
                        f"Conflicting future lengths: {source} has {inferred} "
                        f"but {key} has {length}"
                    )
            # Format 2 key
            fut_cov = item.get("future_covariates")
            if isinstance(fut_cov, dict) and fut_cov:
                for cov_key, cov_val in fut_cov.items():
                    arr = np.asarray(cov_val)
                    if arr.size == 0:
                        continue
                    length = arr.shape[-1]
                    if inferred is None:
                        inferred = length
                        source = f"future_covariates[{cov_key!r}]"
                    elif inferred != length:
                        raise ValueError(
                            f"Conflicting future lengths: {source} has {inferred} "
                            f"but future_covariates[{cov_key!r}] has {length}"
                        )

        if inferred is not None:
            if prediction_length is not None and prediction_length != inferred:
                raise ValueError(
                    f"prediction_length={prediction_length} does not match "
                    f"the length inferred from {source} ({inferred})"
                )
            return inferred

        if prediction_length is not None:
            return prediction_length

        return self.model.iamtime_config.patch_size * self.model.iamtime_config.max_output_steps

    def embed(
        self,
        icl_inputs: list[dict[str, torch.Tensor]],
        prediction_length: Optional[int] = None,
    ) -> Tuple[dict, dict]:
        """
        Get embeddings for the given input time series.

        Parameters
        ----------
        icl_inputs
            Input time series in any of the supported formats
            (see ``predict`` for the full list).
        prediction_length
            The length of the prediction horizon.

        Returns
        -------
        tuple[dict, dict]
            Encoder embeddings and metadata.

            **Important Note:** In the embeddings dict, invalid positions along the patch and variate dimensions are filled with NaN.
            So while using the embeddings, make sure to ignore NaN values or use NaN-safe operations.
            
            The embeddings dict contains:

            - ``example_patched_embeddings``: shape
              ``(batch_size, num_examples, (1+num_example_exog), (example_history_num_patches+example_future_num_patches), d_model)``

            - ``example_tokens_embeddings``: shape
              ``(batch_size, num_examples, num_example_tokens, d_model)``

            - ``query_target_patched_embeddings``: shape
              ``(batch_size, num_variates, (query_target_history_num_patches+query_target_history_num_future_patches), d_model)``
              where ``query_target_history_num_patches`` is the number of patches from the query target
              history and remaining patches correspond to future target patches.

            - ``query_tokens_embeddings``: shape
              ``(batch_size, num_variates, num_query_tokens, d_model)``
              where ``query_start_token_idx`` is the index of the [START] token and
              ``query_mid_token_idx`` is the index of the [MID] token.

            The metadata dict contains:

            - ``query_target_history_loc_scale``: tuple of tensors
              ``(loc, scale)`` each of shape ``(batch_size, num_variates)`` containing the location and scale 
              used to normalize the query target history for each variate.

            - ``query_target_history_num_patches``: int, the number of patches from the query target history.

            - ``query_start_token_idx``: int, the index of the [START] token in ``query_tokens_embeddings``.

            - ``query_mid_token_idx``: int, the index of the [MID] token in ``query_tokens_embeddings``.

        """
        prediction_length = self._resolve_prediction_length(icl_inputs, prediction_length)
        icl_inputs, group_sizes = self._expand_multivariate(icl_inputs)
        batch_icl_inputs = self._process_icl_inputs(icl_inputs, prediction_length)

        model_prediction_length = self.model.iamtime_config.patch_size * self.model.iamtime_config.max_output_steps
        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            raise ValueError(msg)

        with torch.no_grad():
            (
                X_ex, Tok_ex, X_q, Tok_q
            ), (
                M_ex, Mtok_ex, M_q, Mtok_q
            ), (
                query_hist_loc_scale, P_h_q, start_idx, mid_idx
            ) = self.model.encode(
                example_target_histories=batch_icl_inputs["example_target_histories"],
                example_target_futures=batch_icl_inputs["example_target_futures"],
                example_exog_histories=batch_icl_inputs["example_exog_histories"],
                example_exog_futures=batch_icl_inputs["example_exog_futures"],
                query_target_history=batch_icl_inputs["query_target_history"],
                query_exog_history=batch_icl_inputs["query_exog_history"],
                query_exog_future=batch_icl_inputs["query_exog_future"][..., :model_prediction_length],
                prediction_length=prediction_length,
            )

        # NaN out invalid positions so returned embeddings are clean
        X_ex = X_ex.masked_fill(~M_ex.unsqueeze(-1).bool(), float("nan"))
        Tok_ex = Tok_ex.masked_fill(~Mtok_ex.unsqueeze(-1).bool(), float("nan"))
        X_q = X_q.masked_fill(~M_q.unsqueeze(-1).bool(), float("nan"))
        Tok_q = Tok_q.masked_fill(~Mtok_q.unsqueeze(-1).bool(), float("nan"))

        embeddings = {
            "example_patched_embeddings": X_ex, 
            "example_tokens_embeddings": Tok_ex, 
            "query_target_patched_embeddings": X_q[:,:,0,:,:], 
            "query_tokens_embeddings": Tok_q
        }
        metadata = {
            "query_target_history_loc_scale": (
                query_hist_loc_scale[0][:,0,0,0],
                query_hist_loc_scale[1][:,0,0,0],
            ), 
            "query_target_history_num_patches": P_h_q, 
            "query_start_token_idx": start_idx, 
            "query_mid_token_idx": mid_idx
        }

        if any(g > 1 for g in group_sizes):
            embeddings = {
                k: self._regroup_predictions(v, group_sizes)
                for k, v in embeddings.items()
            }

            regrouped = embeddings["example_patched_embeddings"]
            if isinstance(regrouped, torch.Tensor):
                embeddings["example_patched_embeddings"] = regrouped.mean(dim=1)
            else:
                embeddings["example_patched_embeddings"] = [t.mean(dim=0) for t in regrouped]

            regrouped = embeddings["example_tokens_embeddings"]
            if isinstance(regrouped, torch.Tensor):
                embeddings["example_tokens_embeddings"] = regrouped.mean(dim=1)
            else:
                embeddings["example_tokens_embeddings"] = [t.mean(dim=0) for t in regrouped]

            regrouped = embeddings["query_target_patched_embeddings"]
            if isinstance(regrouped, torch.Tensor):
                # Uniform: (batch, n_v, 1, P, d) -> (batch, n_v, P, d)
                embeddings["query_target_patched_embeddings"] = rearrange(
                    regrouped, "b n_v 1 p d -> b (n_v 1) p d"
                )
            else:
                # Mixed: list of (n_v_i, 1, P, d) -> list of (n_v_i, P, d)
                embeddings["query_target_patched_embeddings"] = [t.squeeze(1) for t in regrouped]

            regrouped = embeddings["query_tokens_embeddings"]
            if isinstance(regrouped, torch.Tensor):
                embeddings["query_tokens_embeddings"] = rearrange(
                    regrouped, "b n_v 1 t d -> b (n_v 1) t d"
                )
            else:
                embeddings["query_tokens_embeddings"] = [t.squeeze(1) for t in regrouped]

            metadata["query_target_history_loc_scale"] = (
                self._regroup_predictions(metadata["query_target_history_loc_scale"][0], group_sizes),
                self._regroup_predictions(metadata["query_target_history_loc_scale"][1], group_sizes),
            )

        return embeddings, metadata

    def predict(  # type: ignore[override]
        self,
        icl_inputs: list[dict[str, torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        """
        Get forecasts for the given input time series.

        Parameters
        ---------------------
        icl_inputs
            Input time series. Accepts any of the following formats:

            **Format 1 – ICL-attribute dicts** (list of ``batch_size`` dicts):
                Each dict may contain the following keys (missing keys
                default to empty arrays):

                - ``example_target_histories``:  ``(E, hist_len_i)`` or list
                  of 1-D arrays (variable length per example).
                - ``example_target_futures``:    ``(E, fut_len_i)`` or list.
                - ``example_exog_histories``:    ``(E, M_i, hist_len_i)`` or
                  list of 2-D arrays (variable num exog / length per example).
                - ``example_exog_futures``:      ``(E, M_i, fut_len_i)`` or list.
                - ``query_target_history``:      ``(hist_len,)`` — 1-D.
                - ``query_exog_history``:        ``(M, hist_len)`` — 2-D.
                - ``query_exog_future``:         ``(M, fut_len)`` — 2-D.
                - ``query_target_future``:       ``(fut_len,)`` — 1-D (optional).

                Where ``E`` = number of ICL examples, ``M`` / ``M_i`` = number
                of exogenous series.

            **Format 2 – target + covariates dicts** (list of ``batch_size`` dicts):
                Each dict has:

                - ``"target"``:            ``(hist_len,)`` — 1-D.
                - ``"past_covariates"``:   ``dict[str, array(hist_len,)]`` (optional).
                - ``"future_covariates"``: ``dict[str, array(fut_len,)]`` (optional).

                Covariates with mismatched keys across past/future are
                NaN-padded to maintain aligned indices.

            **Format 3 – 2-D array or tensor** ``(batch_size, hist_len)``:
                Each row is a univariate ``query_target_history`` with no
                examples or exogenous covariates.

            **Format 4 – list of 1-D arrays/tensors** (length ``batch_size``):
                Each element ``(hist_len_i,)`` is a univariate
                ``query_target_history`` (variable length across the batch).

            **Multivariate targets** (all formats):
                In any format, the target may be 2-D
                ``(n_variates, hist_len)`` (and correspondingly
                ``(n_variates, fut_len)`` for the future, if provided).
                For Format 3, pass a 3-D array
                ``(batch_size, n_variates, hist_len)``.  For Format 4,
                each element can be ``(n_variates, hist_len_i)``.

        prediction_length
            The length of the prediction horizon.
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. False by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            - Univariate:
              ``(batch_size, num_quantiles, prediction_length)``.
            - Uniform multivariate (all inputs have the same ``n_variates``):
              ``(batch_size, n_variates, num_quantiles, prediction_length)``.
            - Mixed multivariate (varying ``n_variates``):
              list of length ``batch_size`` where element *i* has shape
              ``(n_variates_i, num_quantiles, prediction_length)``.

        Raises
        ------
        ValueError
            When limit_prediction_length is True and the prediction_length is
            greater than model's training prediction_length.
        """
        prediction_length = self._resolve_prediction_length(icl_inputs, prediction_length)
        icl_inputs, group_sizes = self._expand_multivariate(icl_inputs)
        batch_icl_inputs = self._process_icl_inputs(icl_inputs, prediction_length)

        model_prediction_length = self.model.iamtime_config.patch_size * self.model.iamtime_config.max_output_steps
        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # Unroll the forecast with the full forecast horizon that the 
        # model was trained with. Variance collapses every `horizon` steps.
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    example_target_histories=batch_icl_inputs["example_target_histories"],
                    example_target_futures=batch_icl_inputs["example_target_futures"],
                    example_exog_histories=batch_icl_inputs["example_exog_histories"],
                    example_exog_futures=batch_icl_inputs["example_exog_futures"],
                    query_target_history=batch_icl_inputs["query_target_history"],
                    query_exog_history=batch_icl_inputs["query_exog_history"],
                    query_exog_future=batch_icl_inputs["query_exog_future"][..., :model_prediction_length],
                    prediction_length=(remaining if remaining < model_prediction_length else model_prediction_length),
                ).quantile_preds.to(batch_icl_inputs["query_target_history"])

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]
            center_lower_prediction = prediction[:, central_idx - 1]
            center_upper_prediction = prediction[:, central_idx + 1]
            # Use the mean of the two quantiles around the median as the central prediction to append to the history,
            # since the median can be noisy and cause the history to diverge from the true future
            weighted_central_prediction = (center_lower_prediction + center_upper_prediction) / 2.0
            # weighted_central_prediction = central_prediction

            # - The query's target history is appended with the predicted values. 
            # - The query's history exogenous appends the values from query's future exogenous. 
            #   These values are from the left side of the query's future exogenous, 
            #   and only upto the same length as predictions.
            # - The query's future exogenous is reduced on the left side. 
            #   Reduction is by the same length as predictions.
            query_target_history = batch_icl_inputs["query_target_history"]
            query_target_history = torch.cat([query_target_history, weighted_central_prediction], dim=-1)
            batch_icl_inputs["query_target_history"] = query_target_history

            query_exog_history = batch_icl_inputs["query_exog_history"]
            query_exog_future = batch_icl_inputs["query_exog_future"]
            query_exog_history = torch.cat(
                [query_exog_history, query_exog_future[..., :prediction.shape[-1]]], 
                dim=-1
            )
            batch_icl_inputs["query_exog_history"] = query_exog_history
            batch_icl_inputs["query_exog_future"] = query_exog_future[..., prediction.shape[-1]:]

        flat_result = torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device="cpu"
        )
        if any(g > 1 for g in group_sizes):
            return self._regroup_predictions(flat_result, group_sizes)
        return flat_result

    def predict_quantiles(
        self,
        icl_inputs: list[dict[str, torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get quantile forecasts for the given input time series.

        Parameters
        ---------------------
        icl_inputs
            Input time series in any of the supported formats
            (see ``predict`` for the full list, including multivariate
            targets).
        prediction_length
            The length of the prediction horizon.
        quantile_levels
            The quantile levels to predict.
        **predict_kwargs
            Additional keyword arguments forwarded to ``predict``.

        Returns
        -------
        tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]
            ``(quantiles, mean)``:

            - Univariate:
              ``quantiles``: ``(batch, pred_len, len(quantile_levels))``.
              ``mean``: ``(batch, pred_len)``.
            - Uniform multivariate:
              ``quantiles``: ``(batch, n_variates, pred_len, len(quantile_levels))``.
              ``mean``: ``(batch, n_variates, pred_len)``.
            - Mixed multivariate: lists of per-element tensors.
        """
        # Expand multivariate inputs here so predict sees already-1D targets
        # (predict's own _expand_multivariate will be a no-op on the result).
        expanded_inputs, group_sizes = self._expand_multivariate(icl_inputs)

        # shape (batch_size, prediction_length, len(training_quantile_levels))
        predictions = (
            self.predict(expanded_inputs, prediction_length=prediction_length, **predict_kwargs)
            .detach()
            .swapaxes(1, 2)
        )

        training_quantile_levels = self.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            # no need to perform intra/extrapolation
            quantiles = predictions[
                ..., [training_quantile_levels.index(q) for q in quantile_levels]
            ]
        else:
            # we rely on torch for interpolating quantiles if quantiles that
            # were used in training are not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(
                quantile_levels
            ) > max(training_quantile_levels):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that iAmTime was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which iAmTime "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions,
                q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype),
                dim=-1,
            ).permute(1, 2, 0)
        # NOTE: the median is returned as the mean here
        mean = predictions[:, :, training_quantile_levels.index(0.5)]

        if any(g > 1 for g in group_sizes):
            quantiles = self._regroup_predictions(quantiles, group_sizes)
            mean = self._regroup_predictions(mean, group_sizes)

        return quantiles, mean

    def predict_classes(
        self,
        icl_inputs: list[dict],
        prediction_length: Optional[int] = None,
    ) -> List[dict]:
        """Classify query time series given labelled ICL examples.

        The caller provides **scalar class labels** (not time series) as
        example futures.  This method converts them into pre-compensated
        constant label-code series that survive per-series InstanceNorm
        unchanged, runs the model, and maps the output back to class labels.

        Parameters
        ----------
        icl_inputs
            List of *batch_size* dicts in **Format 1** (ICL-attribute dicts).
            Each dict must contain:

            - ``example_target_histories``: ``(E, hist_len_i)`` or list of
              1-D arrays — historical target series for each example.
            - ``example_target_futures``: 1-D array or list of ``E`` **scalar
              class labels** (int, float, or str), one per example.
            - ``query_target_history``: ``(hist_len,)`` — 1-D target history
              for the series to classify.

            Optional keys (forwarded to the model unchanged):
            ``example_exog_histories``, ``example_exog_futures``,
            ``query_exog_history``, ``query_exog_future``.

        prediction_length
            Length of the synthetic label-code series used internally.
            Defaults to the model's ``patch_size``.  If ``query_exog_future``
            is provided and *prediction_length* is ``None``, its last-axis
            length is used instead.

        Returns
        -------
        list[dict]
            One dict per batch item with:

            - ``"predicted_label"`` (str) — the predicted class.
            - ``"scores"`` (dict[str, float]) — softmax confidence per class.
        """
        from .episode_samplers import _instance_norm_loc_scale

        patch_size = self.model.iamtime_config.patch_size
        model_prediction_length = (
            patch_size * self.model.iamtime_config.max_output_steps
        )

        # ── Determine future-series length ──────────────────────────
        if prediction_length is not None:
            fut_len = prediction_length
        else:
            fut_len = patch_size
            for item in icl_inputs:
                qef = np.asarray(item.get("query_exog_future", []))
                if qef.size > 0:
                    fut_len = qef.shape[-1]
                    break

        # ── Per-item: assign codes, pre-compensate, build series ────
        processed_inputs: List[dict] = []
        batch_meta: List[dict] = []

        for item in icl_inputs:
            item = dict(item)  # shallow copy
            ex_hists_raw = item["example_target_histories"]
            ex_labels_raw = item["example_target_futures"]

            if isinstance(ex_hists_raw, np.ndarray) and ex_hists_raw.ndim == 2:
                ex_hists = [ex_hists_raw[i] for i in range(ex_hists_raw.shape[0])]
            else:
                ex_hists = [np.asarray(h, dtype=np.float64) for h in ex_hists_raw]

            ex_labels = [str(l) for l in np.asarray(ex_labels_raw).ravel()]
            assert len(ex_hists) == len(ex_labels), (
                f"#histories ({len(ex_hists)}) != #labels ({len(ex_labels)})"
            )

            # Deterministic, evenly-spaced label codes in [1, 9]
            unique_labels = sorted(set(ex_labels))
            assert len(unique_labels) >= 2, (
                f"Need >= 2 classes, got {unique_labels}"
            )
            codes = np.linspace(1.0, 9.0, len(unique_labels)).tolist()
            label_codes = dict(zip(unique_labels, codes))

            # Build pre-compensated example futures
            ex_futs = []
            for h, lab in zip(ex_hists, ex_labels):
                loc, scale = _instance_norm_loc_scale(
                    np.asarray(h, dtype=np.float64)
                )
                raw_code = label_codes[lab] * scale + loc
                ex_futs.append(np.full(fut_len, raw_code, dtype=np.float32))

            # Query statistics for back-mapping
            q_hist = np.asarray(item["query_target_history"], dtype=np.float64)
            q_loc, q_scale = _instance_norm_loc_scale(q_hist)

            batch_meta.append({
                "label_codes": label_codes,
                "unique_labels": unique_labels,
                "q_loc": q_loc,
                "q_scale": q_scale,
            })

            item["example_target_futures"] = np.array(ex_futs)
            item.pop("query_target_future", None)
            processed_inputs.append(item)

        # ── Run model ───────────────────────────────────────────────
        batch_icl_inputs = self._process_icl_inputs(processed_inputs, fut_len)

        with torch.no_grad():
            output = self.model(
                example_target_histories=batch_icl_inputs["example_target_histories"],
                example_target_futures=batch_icl_inputs["example_target_futures"],
                example_exog_histories=batch_icl_inputs["example_exog_histories"],
                example_exog_futures=batch_icl_inputs["example_exog_futures"],
                query_target_history=batch_icl_inputs["query_target_history"],
                query_exog_history=batch_icl_inputs["query_exog_history"],
                query_exog_future=batch_icl_inputs["query_exog_future"][
                    ..., :model_prediction_length
                ],
                prediction_length=fut_len,
            )

        # ── Map predictions → class labels ──────────────────────────
        quantile_preds = output.quantile_preds  # (B, Q, pred_len)
        median_idx = int(
            torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
        )
        # Median prediction, averaged over the prediction horizon
        median_preds = quantile_preds[:, median_idx, :].mean(dim=-1)  # (B,)
        pred_raw = median_preds.cpu().numpy()

        results: List[dict] = []
        for i, raw_val in enumerate(pred_raw):
            meta = batch_meta[i]
            labels = meta["unique_labels"]
            lc = meta["label_codes"]

            # Convert prediction from raw space → code space
            pred_code = (float(raw_val) - meta["q_loc"]) / meta["q_scale"]

            # Distance to each class in code space
            dists = np.array([abs(lc[lab] - pred_code) for lab in labels])

            # Softmax confidence (temperature = 1.0 in code space)
            logits = -dists
            exp_logits = np.exp(logits - logits.max())
            probs = exp_logits / exp_logits.sum()

            results.append({
                "predicted_label": labels[int(np.argmin(dists))],
                "scores": {lab: float(p) for lab, p in zip(labels, probs)},
            })

        return results

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """

        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "iamtime_config"), "Not a iAmTime config file"
        assert hasattr(config, "icl_config"), "Not a ICL model config file"

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(
                f"Unknown architecture: {architecture}, defaulting to iAmTimeModel"
            )
            class_ = iAmTimeModel

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)
