"""
Module for calculating satellite visibility, line-of-sight access, and link budget metrics.
"""

import numpy as np
from scipy.interpolate import interp1d, CubicHermiteSpline

# --- Physical Constants ---
SPEED_OF_LIGHT = 299792458.0  # Speed of light [m/s]
BOLTZMANN_CONST = 1.38e-23    # Boltzmann constant [J/K]
R_EQ = 6378e3                 # Earth equatorial radius [m]
R_POL = 6357e3                # Earth polar radius [m]

def sat_visibility(sim_time_s: np.ndarray, rcver_pos: np.ndarray, sat_time_s: np.ndarray, 
                   sat_pos: np.ndarray, sat_vel: np.ndarray, rcver_antenna_dir: np.ndarray, 
                   rcver_max_offb_deg: float, sat_max_offb_deg: float, min_ctnr: float, 
                   rcver_gain_coelev: np.ndarray, sat_eirp_coelev: np.ndarray, 
                   carrier_freq: float) -> tuple:
    """
    Calculates whether a given navigation satellite is accessible by the receiver.

    Visibility restrictions include: Earth obstruction (accounting for oblateness), 
    receiver and transmitter fields of view, and carrier-to-noise ratio (C/N0) requisites.

    Parameters
    ----------
    sim_time_s : numpy.ndarray
        Array of simulation times corresponding to the receiver's trajectory [s], shape (N,).
    rcver_pos : numpy.ndarray
        Receiver position vectors [m], shape (N, 3).
    sat_time_s : numpy.ndarray
        Array of simulation times corresponding to the satellite's ephemeris [s], shape (M,).
    sat_pos : numpy.ndarray
        Navigation satellite position vectors [m], shape (M, 3).
    sat_vel : numpy.ndarray
        Navigation satellite velocity vectors [m/s], shape (M, 3).
    rcver_antenna_dir : numpy.ndarray
        Receiver antenna orientation vector(s), shape (N, 3).
    rcver_max_offb_deg : float
        Maximum off-boresight angle for the receiver antenna [deg].
    sat_max_offb_deg : float
        Maximum off-boresight Tx angle for the navigation satellite antenna [deg].
    min_ctnr : float
        Carrier-to-noise ratio required for data acquisition [dB-Hz].
    rcver_gain_coelev : numpy.ndarray
        2D array mapping receiver antenna gain (column 1) to co-elevation (column 0).
    sat_eirp_coelev : numpy.ndarray
        2D array mapping satellite EIRP (column 1) to co-elevation (column 0).
    carrier_freq : float
        Carrier frequency of the GNSS signal [Hz].

    Returns
    -------
    tuple
        - access (numpy.ndarray): Binary array (1 or 0) indicating overall satellite accessibility.
        - ctnr (numpy.ndarray): Raw computed C/N0 values for all time steps [dB-Hz].
        - ctnr_filtered (numpy.ndarray): C/N0 values only for steps not blocked by Earth and within FOVs.
        - tx_angles_filtered (numpy.ndarray): Transmit angles for accessible steps [deg].
        - rx_angles_filtered (numpy.ndarray): Receive angles for accessible steps [deg].
    """
    
    # 1. Interpolate Satellite Ephemeris to match Receiver Timebase
    sat_pos_interpolator = CubicHermiteSpline(sat_time_s, sat_pos, sat_vel, axis=0)
    sat_pos = sat_pos_interpolator(sim_time_s)

    # 2. Geometric Angles & Vectors
    # Navigation satellite antenna is oriented towards the center of the Earth (Nadir)
    sat_pos_magnitudes = np.linalg.norm(sat_pos, axis=1, keepdims=True)
    sat_antenna_dir = -sat_pos / sat_pos_magnitudes

    # Relative position vector
    rel_pos = sat_pos - rcver_pos
    rel_pos_norm = np.linalg.norm(rel_pos, axis=1)

    # Reception and transmission off-boresight angles
    rx_cos = np.sum(rel_pos * rcver_antenna_dir, axis=1) / rel_pos_norm
    rx_angles = np.rad2deg(np.arccos(np.clip(rx_cos, -1.0, 1.0)))

    tx_cos = np.sum(-rel_pos * sat_antenna_dir, axis=1) / rel_pos_norm
    tx_angles = np.rad2deg(np.arccos(np.clip(tx_cos, -1.0, 1.0)))

    # 3. Link Budget Parameters
    # EIRP Interpolation
    coelev_eirp, eirp = sat_eirp_coelev[:, 0], sat_eirp_coelev[:, 1]
    interp_EIRP = interp1d(coelev_eirp, eirp, kind='linear', bounds_error=False,
                           fill_value=(eirp[0], eirp[-1]))

    # Receiver Gain Interpolation
    coelev_gain, gain = rcver_gain_coelev[:, 0], rcver_gain_coelev[:, 1]
    interp_gain = interp1d(coelev_gain, gain, kind='linear', bounds_error=False, 
                           fill_value=(gain[0], gain[-1]))

    # Signal Attenuation (Free Space Path Loss)
    wavelength = SPEED_OF_LIGHT / carrier_freq
    free_space_loss = -20 * np.log10(wavelength / (4 * np.pi * rel_pos_norm))

    # Thermal Noise
    sys_temp = 290  # System noise temperature [K]
    noise = 10 * np.log10(BOLTZMANN_CONST * sys_temp)
    other_losses = 2  # Receiver implementation losses [dB]

    # Calculate Carrier-to-Noise ratio (assuming zero atmospheric/polarization loss)
    ctnr = interp_EIRP(tx_angles) + interp_gain(rx_angles) - free_space_loss - noise - other_losses

    # 4. Earth Blockage (Accounting for Oblateness)
    rcver_pos_oblate = rcver_pos.copy()
    sat_pos_oblate = sat_pos.copy()
    
    oblateness_ratio = R_EQ / R_POL
    rcver_pos_oblate[:, 2] *= oblateness_ratio
    sat_pos_oblate[:, 2] *= oblateness_ratio
    
    rel_pos_oblate = sat_pos_oblate - rcver_pos_oblate
    R_mask = R_EQ + 1000e3  # 1000 km atmospheric mask
    
    k_earth = -np.sum(rcver_pos_oblate * rel_pos_oblate, axis=1) / np.sum(rel_pos_oblate ** 2, axis=1)
    
    # Point of minimum distance to the Earth's center along the line of sight
    closest_points = rcver_pos_oblate + k_earth[:, np.newaxis] * rel_pos_oblate
    dist_to_center = np.linalg.norm(closest_points, axis=1)
    
    # Boolean masks for visibility logic
    blocked_bool = (k_earth > 0) & (k_earth < 1) & (dist_to_center < R_mask)

    # Overall Access Mask
    access_bool = (~blocked_bool) & (rx_angles <= rcver_max_offb_deg) & \
                  (tx_angles <= sat_max_offb_deg) & (ctnr >= min_ctnr)
    access = access_bool.astype(int) 
    
    # Filtered Data
    mask_ctnr = (~blocked_bool) & (rx_angles <= rcver_max_offb_deg) & (tx_angles <= sat_max_offb_deg)
    ctnr_filtered = ctnr[mask_ctnr]

    mask_angles = (~blocked_bool) & (ctnr >= min_ctnr)
    tx_angles_filtered = tx_angles[mask_angles]
    rx_angles_filtered = rx_angles[mask_angles]

    return access, ctnr, ctnr_filtered, tx_angles_filtered, rx_angles_filtered



