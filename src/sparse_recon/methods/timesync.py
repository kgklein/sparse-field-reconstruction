from sparse_recon.methods.base import ReconstructionMethod


class TimesyncMethod(ReconstructionMethod):
    name = "timesync"

    def fit(self, sample_coords, sample_values):
        raise NotImplementedError(
            "Timesync is a stretch method for later in the project."
        )

    def predict(self, query_coords):
        raise NotImplementedError
