from dataclasses import dataclass
from typing import Optional
from lib.GtEntry import GtEntry

@dataclass
class IJDCalculationEntry:
    gtEntry: GtEntry                        # Reference to the original GT entry
    wzo: int                                # Expectation Compliance Coefficient (WZO)
    visionWeight: Optional[float] = None    # Weight of the vision subsystem (W_WIZ)
    radioWeight: Optional[float] = None     # Weight of the radio subsystem (W_RAD) 
    crad: Optional[int] = None              # Radio Availability Multiplier (C_RAD)
    cwiz: Optional[int] = None              # Visual Detection Correctness Multiplier (C_WIZ)
    visionError: Optional[float] = None     # Calculated euclidean distance of visual detection (E_WIZ)
    radioError: Optional[float] = None      # Calculated euclidean diestance of radio prediction (E_RAD)
    kWiz: Optional[float] = None            # Sensitivity for visual error (kWIZ)
    kRad: Optional[float] = None            # Sensitivity for radio error (kRAD)
    wlr: Optional[float] = None             # Radio Localization Quality Coefficient
    wlw: Optional[float] = None             # Vision Localization Quality Coefficient
    ijd: Optional[float] = None             # Final IJD value