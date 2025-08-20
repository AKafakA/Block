from vidur.types.base_int_enum import BaseIntEnum


def from_value_to_short_metrics_name(value):
    short_cut = {
        1: "min_l",
        2: "max_mbs",
        3: "max_min_bs",
        4: "min_s",
        5: "min_d"
    }
    return short_cut[value]


class TargetMetric(BaseIntEnum):
    """
    Target metrics for the scheduler
    """
    MIN_LATENCY = 1
    MAX_AVG_BATCH_SIZE = 2
    MAX_MIN_BATCH_SIZE = 3
    MIN_SCHEDULING_DELAY = 4
    MIN_DECODING_DELAY = 5
    MIN_GPU_BLOCKS = 6
    MIN_REQUESTS = 7
    MIN_NEW_REQUEST_LATENCY = 8
