"""
Module for simulating GNSS receiver clock bias and drift.
"""

import numpy as np

# Speed of light in m/s
C_LIGHT = 299792458.0  

class ClockSimulator:
    """
    Simulates true clock bias and drift over time based on Allan variance parameters.
    
    The simulation relies on the spectral amplitudes of phase and frequency 
    clock noise. By default, the simulator is initialized with Allan variance 
    parameters representing a standard space-grade OCXO (Oven-Controlled 
    Crystal Oscillator) clock.

    Parameters
    ----------
    h0 : float, optional
        White frequency noise parameter. Default is 2e-25 (OCXO standard).
    h1 : float, optional
        Flicker frequency noise parameter (not directly used in this 2-state model). 
        Default is 7e-25 (OCXO standard).
    h2 : float, optional
        Random walk frequency noise parameter. Default is 6e-25 (OCXO standard).

    Attributes
    ----------
    h0 : float
        Stored white frequency noise parameter.
    h1 : float
        Stored flicker frequency noise parameter.
    h2 : float
        Stored random walk frequency noise parameter.
    Sf : float
        Spectral amplitude of phase noise, derived from h0.
    Sg : float
        Spectral amplitude of frequency noise, derived from h2.
    """

    def __init__(self, h0: float = 2e-25, h1: float = 7e-25, h2: float = 6e-25):
        self.h0 = h0
        self.h1 = h1
        self.h2 = h2
        
        # Calculate spectral amplitudes
        self.Sf = self.h0 / 2.0
        self.Sg = 2 * np.pi**2 * self.h2

    def simulate(self, t_array: np.ndarray, init_bias: float = 0.0, init_drift: float = 0.0) -> tuple:
        """
        Simulates true clock bias and drift using vectorized numpy operations.
        
        Parameters
        ----------
        t_array : numpy.ndarray
            1D array of simulation time epochs [s]. Shape: (N,).
        init_bias : float, optional
            Initial clock bias [s]. Default is 0.0.
        init_drift : float, optional
            Initial clock drift [s/s]. Default is 0.0.
            
        Returns
        -------
        tuple
            - bias (numpy.ndarray): Clock bias state converted to meters. Shape: (N,).
            - drift (numpy.ndarray): Clock drift state converted to meters/second. Shape: (N,).
        """
        dt = np.diff(t_array)
        N_steps = len(dt)
        
        # 1. Vectorized Q_k Construction
        q11 = self.Sf * dt + self.Sg * (dt**3) / 3.0
        q12 = self.Sg * (dt**2) / 2.0
        q22 = self.Sg * dt
        
        Q = np.zeros((N_steps, 2, 2))
        Q[:, 0, 0] = q11
        Q[:, 0, 1] = q12
        Q[:, 1, 0] = q12
        Q[:, 1, 1] = q22
        
        # 2. Vectorized Correlated Noise Generation
        # Cholesky decomposition of all Q matrices simultaneously: Q = L @ L.T
        L = np.linalg.cholesky(Q)
        
        # Generate standard normal noise Z
        Z = np.random.standard_normal((N_steps, 2, 1))
        
        # Apply the Cholesky factor to get correlated noise W = L @ Z
        W = np.matmul(L, Z).reshape(N_steps, 2)
        w_b = W[:, 0]  # Bias noise sequence
        w_d = W[:, 1]  # Drift noise sequence
        
        # 3. Vectorized State Propagation
        # Drift is the cumulative sum of the drift noise
        drift = np.cumsum(w_d) + init_drift
        
        # Bias integrates the previous drift state over dt, plus its own noise
        drift_prev = np.concatenate(([init_drift], drift[:-1]))
        bias_increments = dt * drift_prev + w_b
        bias = np.cumsum(bias_increments) + init_bias
        
        # Prepend the initial states so the arrays perfectly match the length of t_array
        full_bias = np.concatenate(([init_bias], bias))
        full_drift = np.concatenate(([init_drift], drift))
        
        # Convert seconds to meters using the speed of light
        return full_bias * C_LIGHT, full_drift * C_LIGHT
