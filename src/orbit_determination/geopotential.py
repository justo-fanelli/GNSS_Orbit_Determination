"""
Module for calculating the geopotential acceleration using Spherical Harmonics.
"""

import numpy as np
from scipy.special import assoc_legendre_p_all

def gravity_sh_acel(r_ecef: np.ndarray, C: np.ndarray, S: np.ndarray, 
                    mu: float, R: float) -> np.ndarray:
    """
    Computes the gravitational acceleration vector from spherical harmonics.

    Evaluates the gradient of the Earth's geopotential field up to the degree 
    and order specified by the provided coefficient matrices.

    Parameters
    ----------
    r_ecef : numpy.ndarray
        Position vector of the spacecraft in the ECEF frame [m], shape (3,).
    C : numpy.ndarray
        Fully normalized cosine coefficients (C[n,m]).
    S : numpy.ndarray
        Fully normalized sine coefficients (S[n,m]).
    mu : float
        Gravitational parameter of the central body [m^3/s^2].
    R : float
        Reference equatorial radius of the central body [m].

    Returns
    -------
    numpy.ndarray
        Acceleration vector in ECEF frame [m/s^2], shape (3,).
    """
    x, y, z = r_ecef
    r_sq = x**2 + y**2 + z**2
    r = np.sqrt(r_sq)
    xy_dist_sq = x**2 + y**2
    xy_dist = np.sqrt(xy_dist_sq)
    
    # Avoid singularity at the poles
    if xy_dist < 1e-9:
        sphi = 1.0 if z > 0 else -1.0
        cphi = 0.0
        lam = 0.0
    else:
        sphi = z / r
        cphi = xy_dist / r
        lam = np.arctan2(y, x)

    # Max degree/order is derived from the coefficient matrix shape
    N = C.shape[0] - 1 

    # Gravitational potential gradient components
    dU_dr = 0.0
    dU_dphi = 0.0
    dU_dlam = 0.0

    # Associated Legendre polynomials (and derivatives)
    P_all = assoc_legendre_p_all(N + 1, N + 1, sphi, diff_n=1)
    Plm_sin = P_all[0]
    d_Plm_sin = P_all[1]

    # Pre-calculate the (R/r) ratio to avoid division in the inner loop
    R_over_r = R / r
    mu_over_r = mu / r

    # Loop over degree n and order m
    for n in range(2, N + 1):
        # Precompute the radial attenuation factor for degree n
        rho_n = R_over_r**n
        
        for m in range(0, n + 1):
            Cnm = C[n, m]
            Snm = S[n, m]

            # Harmonic terms
            cos_mlam = np.cos(m * lam)
            sin_mlam = np.sin(m * lam)
            common = (Cnm * cos_mlam + Snm * sin_mlam)
            
            # The (-1)^m factor eliminates the Condon-Shortley phase automatically 
            # included by SciPy's function.
            phase = (-1)**m
            
            # Legendre terms with phase cancellation
            # In the new API, degree is axis 0. 
            P_term = Plm_sin[n,m]*phase
            dP_term = d_Plm_sin[n,m]*phase

            # Accumulate partial derivatives
            dU_dr += - (mu_over_r / r) * rho_n * (n + 1) * P_term * common
            dU_dphi += mu_over_r * rho_n * cphi * dP_term * common
            dU_dlam += mu_over_r * rho_n * m * P_term * (Snm * cos_mlam - Cnm * sin_mlam) 

    # Combine partials into Cartesian ECEF acceleration
    mu_r3 = mu / (r_sq * r)
    z_factor = z / (r_sq * xy_dist) if xy_dist > 1e-9 else 0.0
    lam_factor = 1.0 / xy_dist_sq if xy_dist > 1e-9 else 0.0

    a_x = x * (dU_dr / r - dU_dphi * z_factor) - y * (dU_dlam * lam_factor) - mu_r3 * x
    a_y = y * (dU_dr / r - dU_dphi * z_factor) + x * (dU_dlam * lam_factor) - mu_r3 * y
    a_z = z * (dU_dr / r) + dU_dphi * (xy_dist / r_sq) - mu_r3 * z

    return np.array([a_x, a_y, a_z])
