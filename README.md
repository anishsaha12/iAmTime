# iAmTime

**A Foundation Model for Instruction-Conditioned In-Context Time Series Tasks**

iAmTime is a time-series foundation model that adapts at inference time from structured input-output demonstrations. It can combine target histories, known past and future covariates, and completed support examples without updating model parameters.

Paper: [arXiv:2603.22586](https://arxiv.org/abs/2603.22586)

## Checkpoint availability

iAmTime is designed for four task families:

- Forecasting
- Imputation
- Anomaly detection
- Classification

At present, **only the forecasting checkpoint is publicly available** on Hugging Face:

| Checkpoint | Task | Status |
| --- | --- | --- |
| [`anishsaha/iAmTime-base-forecast`](https://huggingface.co/anishsaha/iAmTime-base-forecast) | Forecasting | Available |
| Imputation | Imputation | Not yet released |
| Anomaly detection | Anomaly detection | Not yet released |
| Classification | Classification | Not yet released |

The installation and usage examples below therefore cover forecasting only. The other task checkpoints and their task-specific usage will be documented when they are released.

## Installation

iAmTime requires Python 3.10 or later. Install the package from the repository root:

```bash
pip install .
```

To run the usage notebook, install the notebook dependencies as well:

```bash
pip install ".[notebook]"
```

For development from a clone, use an editable install:

```bash
pip install -e ".[notebook]"
```

## Quick start

```python
import numpy as np
import torch

from iAmTime.pipeline import iAmTimePipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
	torch.bfloat16
	if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
	else torch.float32
)

pipeline = iAmTimePipeline.from_pretrained(
	"anishsaha/iAmTime-base-forecast",
	device_map=device,
	torch_dtype=dtype,
)

history = np.sin(np.arange(192, dtype=np.float32) * 2 * np.pi / 24)
quantiles, median = pipeline.predict_quantiles(
	history[None, :],
	prediction_length=48,
	quantile_levels=[0.1, 0.5, 0.9],
)

print(quantiles.shape)  # (batch=1, horizon=48, quantiles=3)
print(median.shape)     # (batch=1, horizon=48)
```

`from_pretrained` downloads the configuration and weights from the Hugging Face Hub on first use and caches them locally.

## Forecasting inputs

The forecasting pipeline supports several levels of context through the same `predict` and `predict_quantiles` methods.

### Univariate

Pass a 2-D array with shape `(batch, history)`, or a list of 1-D arrays for variable-length histories.

```python
quantiles, median = pipeline.predict_quantiles(
	histories,
	prediction_length=48,
	quantile_levels=[0.1, 0.5, 0.9],
)
```

### Multivariate

Pass an array with shape `(batch, variates, history)`. Each variate is forecast while the others are used as context.

```python
quantiles, median = pipeline.predict_quantiles(
	multivariate_histories,
	prediction_length=48,
	quantile_levels=[0.1, 0.5, 0.9],
)
```

For a uniform batch, quantiles have shape `(batch, variates, horizon, quantiles)`.

### Known covariates

Use dictionaries when past or future-known drivers are available. Past covariates align with the target history; future covariates align with the forecast horizon.

```python
inputs = [{
	"target": target_history,
	"past_covariates": {
		"temperature": past_temperature,
		"promotion": past_promotion,
	},
	"future_covariates": {
		"temperature": future_temperature,
		"promotion": future_promotion,
	},
}]

quantiles, median = pipeline.predict_quantiles(
	inputs,
	prediction_length=len(future_temperature),
	quantile_levels=[0.1, 0.5, 0.9],
)
```

### In-context examples

Completed support examples can demonstrate the relationship the model should apply to a query. The query's target future must not be included at inference time.

```python
inputs = [{
	"example_target_histories": [support_1_history, support_2_history],
	"example_target_futures": [support_1_future, support_2_future],
	"example_exog_histories": [support_1_covariates, support_2_covariates],
	"example_exog_futures": [support_1_future_covariates, support_2_future_covariates],
	"query_target_history": query_history,
	"query_exog_history": query_covariates,
	"query_exog_future": query_future_covariates,
}]

quantiles, median = pipeline.predict_quantiles(
	inputs,
	prediction_length=query_future_covariates.shape[-1],
	quantile_levels=[0.1, 0.5, 0.9],
)
```

See [`notebooks/iamtime_forecasting_usage.ipynb`](notebooks/iamtime_forecasting_usage.ipynb) for complete synthetic examples covering all four forecasting input styles.

## Model overview

iAmTime represents an episode as a structured prompt containing support examples and a query. The implementation:

1. Normalizes and divides each time series into patches.
2. Uses semantic tokens to identify target history, exogenous history, known future covariates, and demonstrated target futures.
3. Applies a Hierarchical Multi-Scope Transformer Encoder to model temporal structure, covariate relationships, and information shared across examples.
4. Uses a Task-Conditioned Patch Decoder with expert routing to produce probabilistic quantile predictions.

This instruction-conditioned design lets one architecture infer the task represented by demonstrations. The currently released checkpoint is trained and exposed for forecasting.

## Citation

```bibtex
@article{saha2026iamtime,
  title={A Foundation Model for Instruction-Conditioned In-Context Time Series Tasks},
  author={Saha, Anish and Shmakov, Konstantin},
  journal={arXiv preprint arXiv:2603.22586},
  year={2026}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).