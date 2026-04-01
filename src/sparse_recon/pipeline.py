from sparse_recon.types import ReconstructionResult
from sparse_recon.metrics.errors import rmse, relative_l2


def run_reconstruction(field, samples, method):
    method.fit(samples.coords, samples.values)
    pred = method.predict(field.coords)

    metrics = {
        "rmse": rmse(field.values, pred),
        "relative_l2": relative_l2(field.values, pred),
    }

    return ReconstructionResult(
        method=method.name,
        query_coords=field.coords,
        predicted_values=pred,
        metrics=metrics,
        metadata={},
    )
