"""
Process data module for geotechnical model response processing.

This module provides functions to process velocity response data and compute
key metrics including V_y,max, V_eff,max, PSD,max, and Freq_PSD,max.
"""

import numpy as np
import SignalProcessingTools.time_signal as time_signal  # type: ignore[import]
from SignalProcessingTools.time_signal import Windows  # type: ignore[import]


def process_response_data(time: np.ndarray, velocity_y: np.ndarray, 
                          window=Windows.HAMMING, window_size: int = 2000) -> dict:
    """
    Process velocity response data to compute key metrics without mdpa/yml dependencies.
    
    This function processes time and velocity data to compute:
    - V_y,max: Maximum absolute velocity in Y direction (mm/s)
    - V_eff,max: Maximum effective velocity (mm/s)
    - PSD,max: Maximum power spectral density ((mm/s)^2/Hz)
    - Freq_PSD,max: Frequency at which PSD is maximum (Hz)
    
    Parameters:
    -----------
    time : np.ndarray
        Time array in seconds
    velocity_y : np.ndarray
        Velocity in Y direction (m/s)
    window : Windows
        Window function for signal processing (default: HAMMING)
    window_size : int
        Window size for signal processing (default: 2000)
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'V_y_max': Maximum absolute velocity Y (mm/s)
        - 'V_eff_max': Maximum effective velocity (mm/s)
        - 'PSD_max': Maximum PSD ((mm/s)^2/Hz)
        - 'Freq_PSD_max': Frequency at max PSD (Hz)
        - 'time': Time array
        - 'velocity_y': Velocity Y array (mm/s)
        - 'v_eff': Effective velocity array (mm/s)
        - 'frequency_Pxx': Frequency array for PSD (Hz)
        - 'Pxx': PSD array ((mm/s)^2/Hz)
    """
    # Convert velocity_y to numpy array if needed
    velocity_y = np.asarray(velocity_y, dtype=float)
    time = np.asarray(time, dtype=float)
    
    # Process the time signal
    signal = time_signal.TimeSignalProcessing(
        time,
        velocity_y,
        window=window,
        window_size=window_size
    )
    signal.psd()
    signal.v_eff_SBR()
    
    # Compute metrics
    # V_y,max: maximum absolute velocity in mm/s
    v_y_max = np.max(np.abs(velocity_y * 1000))
    
    # V_eff,max: maximum effective velocity in mm/s
    v_eff_max = np.max(signal.v_eff)
    
    # PSD,max: maximum PSD in (mm/s)^2/Hz
    psd_max = np.max(signal.Pxx) * 1000**2
    
    # Freq_PSD,max: frequency at which PSD is maximum
    freq_psd_max = signal.frequency_Pxx[np.argmax(signal.Pxx)]
    
    # Prepare return dictionary
    result = {
        'V_y_max': float(v_y_max),
        'V_eff_max': float(v_eff_max),
        'PSD_max': float(psd_max),
        'Freq_PSD_max': float(freq_psd_max),
        'time': time,
        'velocity_y': velocity_y * 1000,  # Convert to mm/s
        'v_eff': signal.v_eff[:len(time)] if len(signal.v_eff) >= len(time) else signal.v_eff,
        'frequency_Pxx': signal.frequency_Pxx,
        'Pxx': signal.Pxx * 1000**2  # Convert to (mm/s)^2/Hz
    }
    
    return result


#| PSD Peak Range | Possible Source                         |
#| -------------- | --------------------------------------- |
#| < 5 Hz         | Soil waves / ground resonance           |
#| 5–20 Hz        | Building annoyance zone                 |
#| 20–40 Hz       | Structural resonances                   |
#| 40–80 Hz       | Wheel–rail interaction (axles, defects) |
#| >100 Hz        | Impacts + mechanical defects            |


#| Zone       | v_eff range    | Meaning                           |
#| ---------- | -------------- | --------------------------------- |
#| **Green**  | < 0.1 mm/s     | No concern / scarcely perceptible |
#| **Yellow** | 0.1 – 0.3 mm/s | Noticeable; *monitor if frequent* |
#| **Red**    | > 0.3 mm/s     | Disturbing / risk of complaints   |

#Healthcare criteria
#| Period  | Threshold `V_eff,max(30 s)` | Max allowed number of intervals |
#| ------- | --------------------------- | ------------------------------- |
#| Day     | 0.4 mm/s                    | limited (e.g. 22 intervals)     |
#| Evening | 0.4 mm/s                    | stricter than day               |
#| Night   | **0.2 mm/s**                | stricter, many fewer allowed    |
