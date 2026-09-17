"""Built-in adapter registrations."""

from .audio_imu_cough import AudioImuCoughAdapter
from .hhar import HharAdapter
from .image_benchmarks import Cifar10Adapter, FashionMnistAdapter
from .pads import PadsAdapter
from .pulse_transit_time import PulseTransitTimeAdapter
from .ringauth import RingAuthAdapter
from .sisfall import SisFallAdapter
from .uci_har import UciHarAdapter
from .wesad import WesadAdapter

__all__ = ["AudioImuCoughAdapter", "Cifar10Adapter", "FashionMnistAdapter", "HharAdapter", "PadsAdapter", "PulseTransitTimeAdapter", "RingAuthAdapter", "SisFallAdapter", "UciHarAdapter", "WesadAdapter"]
