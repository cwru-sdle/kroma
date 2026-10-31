"""In-process listener for token counts and refinement timings."""


class MetricsListener:
    """Collects API counters and refinement timings for one run.

    This is a local observer for the matching loop. It does not record
    conversation traces.

    Attributes:
        api (dict[str, int]): ``input_token``, ``output_token``,
            ``token_count`` and ``api_call_cnt`` totals.
        timings (dict[str, float]): Seconds spent per bucket, such as
            ``offline_refinement`` and ``online_refinement``.
    """

    def __init__(self):
        self.api = {
            "input_token": 0,
            "output_token": 0,
            "token_count": 0,
            "api_call_cnt": 0,
        }
        self.timings = {
            "offline_refinement": 0.0,
            "online_refinement": 0.0,
        }

    def add_api(self, metrics: dict) -> None:
        """Add one call's counters to the running totals.

        Args:
            metrics (dict): Counters keyed like :attr:`api`; missing keys
                count as zero and unknown keys are ignored.
        """
        for key in self.api:
            self.api[key] += metrics.get(key, 0)

    def add_timing(self, bucket: str, seconds: float) -> None:
        """Add elapsed time to a timing bucket, creating it if needed.

        Args:
            bucket (str): Bucket name.
            seconds (float): Seconds to add.
        """
        self.timings[bucket] = self.timings.get(bucket, 0.0) + seconds
