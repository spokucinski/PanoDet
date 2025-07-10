from dataclasses import dataclass
from typing import Optional
from lib.GtEntry import GtEntry

@dataclass
class CalculationEntry:
    gtEntry: GtEntry                                        # Reference to the original GT entry
    ECC: int                                                # Expectation Compliance Coefficient (ECC)
    visionWeight: Optional[float] = None                    # Weight of the vision subsystem (W_VIS)
    radioWeight: Optional[float] = None                     # Weight of the radio subsystem (W_RAD) 
    crad: Optional[int] = None                              # Radio Availability Multiplier (C_RAD)
    cwiz: Optional[int] = None                              # Visual Detection Correctness Multiplier (C_VIS)
    visionError: Optional[float] = None                     # Calculated euclidean distance of visual detection (E_VIS)
    radioError: Optional[float] = None                      # Calculated euclidean diestance of radio prediction (E_RAD)
    visionSensitivity: Optional[float] = None               # Sensitivity for visual error (kVIS)
    radioSensitivity: Optional[float] = None                # Sensitivity for radio error (kRAD)
    wlr: Optional[float] = None                             # Radio Localization Quality Coefficient
    wlw: Optional[float] = None                             # Vision Localization Quality Coefficient
    ijd: Optional[float] = None                             # Final IJD value
    useHybridPositioning: bool = False                      # Use radio-based positions of cameras (hybrid-mode) in visual error calculation