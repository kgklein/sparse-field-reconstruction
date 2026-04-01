from __future__ import annotations

from sparse_recon.metrics.errors import prediction_validity_summary, relative_l2, rmse
from sparse_recon.metrics.physics import divergence_rmse_2d
from sparse_recon.sampling.sampler import sample_field
from sparse_recon.types import FieldSnapshot, ReconstructionResult, SampleSet


def evaluate_reconstruction(
    field: FieldSnapshot,
    pred,
    *,
    include_divergence: bool = True,
) -> dict:
    metrics = {
        "rmse": rmse(field.values, pred),
        "relative_l2": relative_l2(field.values, pred),
    }
    metrics.update(prediction_validity_summary(pred))

    is_structured_2d = (
        include_divergence
        and field.grid_shape is not None
        and field.axes is not None
        and len(field.grid_shape) == 2
        and field.values.shape[1] == 2
        and "x" in field.axes
        and "y" in field.axes
    )
    if is_structured_2d:
        metrics["divergence_rmse"] = divergence_rmse_2d(
            field.values,
            pred,
            field.grid_shape,
            field.axes["x"],
            field.axes["y"],
        )

    return metrics


def run_reconstruction(
    field: FieldSnapshot,
    samples: SampleSet,
    method,
    *,
    include_divergence: bool = True,
    metadata: dict | None = None,
) -> ReconstructionResult:
    method.fit(samples.coords, samples.values)
    pred = method.predict(field.coords)
    metrics = evaluate_reconstruction(
        field,
        pred,
        include_divergence=include_divergence,
    )

    result_metadata = {
        "field": dict(field.metadata or {}),
        "sampling": dict(samples.metadata or {}),
        "method": {
            "name": method.name,
            "params": method.get_params(),
        },
    }
    if metadata:
        result_metadata.update(metadata)

    return ReconstructionResult(
        method=method.name,
        query_coords=field.coords,
        predicted_values=pred,
        metrics=metrics,
        metadata=result_metadata,
    )


def run_sampling_experiment(
    field: FieldSnapshot,
    sample_coords,
    method,
    *,
    noise_sigma: float = 0.0,
    noise_seed: int = 0,
    include_divergence: bool = True,
    metadata: dict | None = None,
) -> tuple[SampleSet, ReconstructionResult]:
    samples = sample_field(
        field,
        sample_coords,
        noise_sigma=noise_sigma,
        seed=noise_seed,
    )
    result = run_reconstruction(
        field,
        samples,
        method,
        include_divergence=include_divergence,
        metadata=metadata,
    )
    return samples, result
