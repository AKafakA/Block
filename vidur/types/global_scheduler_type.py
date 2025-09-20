from vidur.types.base_int_enum import BaseIntEnum


class GlobalSchedulerType(BaseIntEnum):
    RANDOM = 1
    ROUND_ROBIN = 2
    LOR = 3
    LODT = 4
    # opt global scheduler name
    OPT = 5
    MIN_MEMORY = 6
    BLOCK_OFFLINE = 7
    BLOCK_STAR_OFFLINE = 8
    INFAAS_PLUS_PLUS = 9
    LLUMNIX_MINUS = 10
