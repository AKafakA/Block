from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int


@dataclass
class A30DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 165
    total_memory_gb: int = 24

    @staticmethod
    def get_type():
        return DeviceSKUType.A30


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H100


@dataclass
class A100_40GBDeviceSKUConfig(BaseDeviceSKUConfig):
    """CloudLab d8545: NVIDIA A100 SXM4 40GB

    Specs:
    - 40GB HBM2e memory
    - 1.6 TB/s memory bandwidth (vs 2.0 TB/s on 80GB)
    - 312 TFLOPS FP16 (same as 80GB)
    - NVLink 600 GB/s (same as 80GB)
    """
    fp16_tflops: int = 312
    total_memory_gb: int = 40

    @staticmethod
    def get_type():
        return DeviceSKUType.A100_40GB
