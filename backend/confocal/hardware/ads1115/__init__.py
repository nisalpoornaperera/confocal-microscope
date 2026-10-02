"""ADS1115 16-bit ADC reading the OPT101 photodiode amplifier.

* :mod:`.conversion` - code <-> voltage conversion and PGA range selection (pure);
* :mod:`.adc` - ``ADS1115ADC``, the I2C driver (smbus2 register access, the
  blocking bus calls run in ``asyncio.to_thread``).
"""

from confocal.hardware.ads1115.adc import (
    ADS1115ADC,
    BusFactory,
    ConversionMode,
    SMBusLike,
    encode_config,
    mux_code,
    open_smbus,
)
from confocal.hardware.ads1115.conversion import (
    CODE_MAX,
    CODE_MIN,
    GAINS_HIGH_TO_LOW,
    VALID_DATA_RATES,
    conversion_time_s,
    counts_to_volts,
    is_saturated,
    select_gain,
    volts_to_counts,
)

__all__ = [
    "ADS1115ADC",
    "CODE_MAX",
    "CODE_MIN",
    "GAINS_HIGH_TO_LOW",
    "VALID_DATA_RATES",
    "BusFactory",
    "ConversionMode",
    "SMBusLike",
    "conversion_time_s",
    "counts_to_volts",
    "encode_config",
    "is_saturated",
    "mux_code",
    "open_smbus",
    "select_gain",
    "volts_to_counts",
]
