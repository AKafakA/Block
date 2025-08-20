from vidur.types.base_int_enum import BaseIntEnum


class GlobalSchedulerType(BaseIntEnum):
    RANDOM = 1
    ROUND_ROBIN = 2
    LOR = 3
    LODT = 4
    # opt global scheduler name
    OPT = 5
    MIN_MEMORY = 6
