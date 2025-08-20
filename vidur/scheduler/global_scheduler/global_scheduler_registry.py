from vidur.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler
from vidur.scheduler.global_scheduler.min_memory_scheduler import MinMemoryScheduler
from vidur.scheduler.global_scheduler.random_global_scheduler import (
    RandomGlobalScheduler,
)
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import (
    RoundRobinGlobalScheduler,
)
from vidur.scheduler.global_scheduler.lodt_scheduler import LODTScheduler
from vidur.scheduler.global_scheduler.length_aware_optimal_scheduler import LengthAwareOptimalScheduler
from vidur.types import GlobalSchedulerType
from vidur.utils.base_registry import BaseRegistry


class GlobalSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> GlobalSchedulerType:
        return GlobalSchedulerType.from_str(key_str)


GlobalSchedulerRegistry.register(GlobalSchedulerType.RANDOM, RandomGlobalScheduler)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.ROUND_ROBIN, RoundRobinGlobalScheduler
)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR, LORGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.MIN_MEMORY, MinMemoryScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LODT, LODTScheduler)

GlobalSchedulerRegistry.register(GlobalSchedulerType.OPT, LengthAwareOptimalScheduler)

