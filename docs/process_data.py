"""
Process data module for geotechnical model response processing.

Computes V_y,max, V_eff,max, PSD,max, Freq_PSD,max, and three-band V_eff
following Chen et al. (2014, JVE 16/4) - no IBS dependency.

Band V_eff is derived by integrating the one-sided PSD over each frequency
range and taking the square root (Chen et al. Eq. 2-3):

    Low     1 -   8 Hz   soil / ground-wave propagation   (lowest attenuation)
    Mid    10 -  25 Hz   building / structural annoyance
    High  31.5 - 100 Hz  track / wheel-rail interaction   (highest attenuation)

Gaps at 8-10 Hz and 25-31.5 Hz are intentional (match the paper grouping).
"""

import numpy as np

try:
    import SignalProcessingTools.time_signal as time_signal  # type: ignore[import]
    from SignalProcessingTools.time_signal import Windows  # type: ignore[import]
except ModuleNotFoundError:
    from scipy.signal import welch

    class Windows:
        HAMMING = "hamming"

    time_signal = None  # type: ignore[assignment]
else:
    from scipy.signal import welch  # noqa: F401


# Frequency bands — Chen et al. (2014), Section 5 / Table 2
FREQ_BANDS = {
    'V_eff_low':  (1.0,   8.0),
    'V_eff_mid':  (10.0,  25.0),
    'V_eff_high': (31.5, 100.0),
}


def compute_band_v_eff(freq: np.ndarray, psd_mm2: np.ndarray) -> dict:
    """
    V_eff (mm/s) in each of the three frequency bands via PSD integration.

    Implements Chen et al. (2014) Eq. 2-3:
        E_y(band) = integral[f_l -> f_u]  S_y(f) df        [(mm/s)^2]
        V_eff(band) = sqrt(E_y(band))                       [mm/s]

    Parameters
    ----------
    freq    : 1-D frequency array (Hz) from one-sided PSD
    psd_mm2 : one-sided PSD in (mm/s)^2 / Hz

    Returns
    -------
    dict  {'V_eff_low': float, 'V_eff_mid': float, 'V_eff_high': float}
          NaN when the band has fewer than 2 PSD bins (Nyquist below band limit).
    """
    out = {}
    nyq = float(freq[-1]) if len(freq) > 0 else 0.0
    for key, (f_lo, f_hi) in FREQ_BANDS.items():
        if f_hi > nyq:
            out[key] = np.nan
            continue
        mask = (freq >= f_lo) & (freq <= f_hi)
        if np.sum(mask) < 2:
            out[key] = np.nan
        else:
            energy = float(np.trapz(psd_mm2[mask], freq[mask]))
            out[key] = float(np.sqrt(max(energy, 0.0)))
    return out


def process_response_data(time: np.ndarray, velocity_y: np.ndarray,
                          window=Windows.HAMMING, window_size: int = 2000) -> dict:
    """
    Process velocity time series and compute key vibration metrics.

    Parameters
    ----------
    time       : time array (s)
    velocity_y : velocity in Y direction (m/s)
    window     : window function (default: HAMMING)
    window_size: window length for PSD (default: 2000)

    Returns
    -------
    dict with keys:
        V_y_max      -- peak absolute velocity Y (mm/s)
        V_eff_max    -- peak effective (RMS) velocity, SBR-style (mm/s)
        PSD_max      -- peak one-sided PSD ((mm/s)^2/Hz)
        Freq_PSD_max -- frequency at PSD peak (Hz)
        V_eff_low    -- V_eff integrated over  1 -   8 Hz (mm/s)
        V_eff_mid    -- V_eff integrated over 10 -  25 Hz (mm/s)
        V_eff_high   -- V_eff integrated over 31.5-100 Hz (mm/s)
        time         -- time array (s)
        velocity_y   -- velocity Y (mm/s)
        v_eff        -- time-varying effective velocity (mm/s)
        frequency_Pxx-- frequency array (Hz)
        Pxx          -- one-sided PSD ((mm/s)^2/Hz)
    """
    velocity_y = np.asarray(velocity_y, dtype=float)
    time = np.asarray(time, dtype=float)

    # SignalProcessingTools requires window_size <= signal length -- clamp defensively.
    effective_window = min(window_size, len(velocity_y))

    if time_signal is not None:
        signal = time_signal.TimeSignalProcessing(
            time,
            velocity_y,
            window=window,
            window_size=effective_window,
        )
        signal.psd()
        signal.v_eff_SBR()
    else:
        dt = float(time[1] - time[0]) if len(time) > 1 else 0.0
        if dt <= 0:
            raise ValueError("Invalid time vector for PSD computation.")
        fs = 1.0 / dt
        nperseg = effective_window
        frequency_Pxx, Pxx = welch(
            velocity_y,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=max(0, nperseg // 2),
            detrend=False,
            scaling="density",
        )
        v_rms = float(np.sqrt(np.trapz(Pxx, frequency_Pxx)))  # m/s

        class _FallbackSignal:
            pass

        signal = _FallbackSignal()
        signal.frequency_Pxx = frequency_Pxx
        signal.Pxx = Pxx
        signal.v_eff = np.full_like(time, v_rms * 1000.0, dtype=float)  # mm/s

    # Broadband metrics
    # V_y_max = max|v_y(t)| over the full time history — peak occurs when the moving load
    # is closest to this point (different t for each grid location along the track).
    v_y_max      = float(np.max(np.abs(velocity_y * 1000)))
    v_eff_max    = float(np.max(signal.v_eff))
    psd_mm2_arr  = signal.Pxx * 1e6           # (m/s)^2/Hz -> (mm/s)^2/Hz
    psd_max      = float(np.max(psd_mm2_arr))
    freq_psd_max = float(signal.frequency_Pxx[np.argmax(psd_mm2_arr)])

    # Band V_eff via PSD integration (Chen et al. 2014 Eq. 2-3)
    bands = compute_band_v_eff(signal.frequency_Pxx, psd_mm2_arr)

    return {
        'V_y_max':       v_y_max,
        'V_eff_max':     v_eff_max,
        'PSD_max':       psd_max,
        'Freq_PSD_max':  freq_psd_max,
        **bands,                              # V_eff_low, V_eff_mid, V_eff_high
        'time':          time,
        'velocity_y':    velocity_y * 1000,   # mm/s
        'v_eff':         signal.v_eff[:len(time)] if len(signal.v_eff) >= len(time) else signal.v_eff,
        'frequency_Pxx': signal.frequency_Pxx,
        'Pxx':           psd_mm2_arr,
    }
