"""Built-in adapter registrations."""

from .audio_imu_cough import AudioImuCoughAdapter
from .hhar import HharAdapter
from .pads import PadsAdapter
from .pulse_transit_time import PulseTransitTimeAdapter
from .ringauth import RingAuthAdapter
from .sisfall import SisFallAdapter
from .uci_har import UciHarAdapter
from .wesad import WesadAdapter

__all__ = ["AudioImuCoughAdapter", "HharAdapter", "PadsAdapter", "PulseTransitTimeAdapter", "RingAuthAdapter", "SisFallAdapter", "UciHarAdapter", "WesadAdapter"]
