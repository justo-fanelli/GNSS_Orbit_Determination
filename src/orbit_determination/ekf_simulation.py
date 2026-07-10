import numpy as np
from orbit_determination.ekf_environment import Environment
from orbit_determination.ekf_spacecraft import Spacecraft
from orbit_determination.ekf_gnss import NavSatSystems
from orbit_determination.ekf import OrbitalEKF

class SimulationManager:
    """
    Orchestrates the asynchronous simulation loop for the Orbit Determination pipeline.

    This manager synchronizes the high-frequency integration of the EKF, the 
    evaluation of the truth environment, the generation of noisy GNSS measurements 
    at discrete epochs, and the logging of state estimates. 

    Parameters
    ----------
    kf : OrbitalEKF
        Initialized instance of the Extended Kalman Filter.
    env : Environment
        Initialized physical environment manager (gravity, third-bodies).
    sat : Spacecraft
        Initialized spacecraft containing the truth trajectory and properties.
    gnss : NavSatSystems
        Initialized GNSS manager with loaded and pre-aligned ephemeris data.
    t_log_array : numpy.ndarray
        Array of time epochs where the EKF state should be logged/saved [s].
    t_meas_array : numpy.ndarray
        Array of time epochs where GNSS measurements are scheduled to occur [s].

    Attributes
    ----------
    kf : OrbitalEKF
        The active Extended Kalman Filter.
    env : Environment
        The active Environment manager.
    sat : Spacecraft
        The active Spacecraft truth model.
    gnss : NavSatSystems
        The active GNSS measurement simulator.
    t_log_array : numpy.ndarray
        The designated logging epochs.
    t_meas_array : numpy.ndarray
        The designated measurement epochs.
    n_log_steps : int
        Total number of logging steps to be executed.
    est_states : numpy.ndarray
        Pre-allocated array for logged state estimates. Shape: (n_log_steps, 8).
    est_covariances : numpy.ndarray
        Pre-allocated array for logged covariance matrices. Shape: (n_log_steps, 8, 8).
    mahalanobis : list of list
        List storing the filter innovation consistency checks. 
        Each entry is [time, degrees_of_freedom, squared_mahalanobis_distance].
    """

    def __init__(self, kf: OrbitalEKF, env: Environment, sat: Spacecraft, gnss: NavSatSystems,
                 t_log_array: np.ndarray, t_meas_array: np.ndarray):
        self.kf = kf
        self.env = env
        self.sat = sat
        self.gnss = gnss
        self.t_log_array = t_log_array  # The user-provided time grid for logging/output
        self.t_meas_array = t_meas_array # The user-provided time grid for measurement updates
        
        # Storage for results (aligned with t_log_array)
        self.n_meas_steps = len(t_meas_array)
        self.n_log_steps = len(t_log_array)
        self.est_states = np.full((self.n_log_steps, 8), np.nan)
        self.est_covariances = np.full((self.n_log_steps, 8, 8), np.nan)
        self.mahalanobis = []

    def _calculate_residual_and_jacobian(self, t: float, rx_estimated_state: np.ndarray, meas_idx: int) -> tuple:
        """
        Computes the measurement innovation (residual) and the observation Jacobian.

        Compares the noisy simulated GNSS measurement from the truth state against 
        the theoretical measurement predicted by the EKF's current estimated state.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        rx_estimated_state : numpy.ndarray
            The EKF's current estimated 8-state vector.
        meas_idx : int
            The integer index corresponding to the current measurement epoch.

        Returns
        -------
        tuple
            - y_res (numpy.ndarray): The measurement innovation vector (z_meas - z_pred). Shape: (M,).
            - H (numpy.ndarray): The observation Jacobian matrix. Shape: (M, 8).
            - R (numpy.ndarray): The measurement noise covariance matrix. Shape: (M, M).
            Returns (None, None, None) if no valid measurements are available at this epoch.
        """
        # Get Truth (The "Sensor Reading")
        # We need to construct the full 8-element truth vector for the helper
        rx_true_state = self.sat.get_true_state(t)
        z_meas, R = self.gnss.get_measurement(meas_idx, rx_true_state)
        
        if z_meas is None:
            return None, None, None
            
        # Get Prediction (The "Observation Model" h(x))
        # This returns the Jacobian H evaluated at the ESTIMATE, which is required for EKF.
        z_pred, H, _ = self.gnss.compute_gnss_observables(meas_idx, rx_estimated_state)
        
        # Calculate Innovation / Residual
        v = z_meas - z_pred
        
        return v, H, R
        
    def run(self) -> tuple:
        """
        Executes the main orbit determination simulation loop.

        Advances time between logging events and measurement events. 
        Propagates the EKF forward, triggers measurement updates when scheduled, 
        evaluates Mahalanobis distances, and records data.

        Returns
        -------
        tuple
            - est_states (numpy.ndarray): Array of all logged state estimates.
            - est_covariances (numpy.ndarray): Array of all logged covariance matrices.
            - mahalanobis (list): The Mahalanobis distance history for filter consistency analysis.
        """

        print("Starting Simulation...")
        t_current = self.t_log_array[0]
        t_final = self.t_log_array[-1]
        
        # Index to track where we are in the t_array (Log) array
        log_idx = 0
        # Log initial state 
        self._log_state(log_idx)
        log_idx += 1
        # Index to track where we are in the measurement time array
        meas_idx = 1

        # Advance measurement time to the t_log_array starting time (t_current)
        while meas_idx < len(self.t_meas_array) and self.t_meas_array[meas_idx] < t_current:
            meas_idx += 1
        
        # Main loop
        while t_current < t_final:

            # Get the next logging time 
            next_log_time = self.t_log_array[log_idx]

            # Get the next measurement time
            if meas_idx < self.n_meas_steps:
                next_meas_time = self.t_meas_array[meas_idx]
            else:
                # if there are no more measurements, we push the next measurement
                # to infinity.
                next_meas_time = t_final + 1.0 
                
            # Determine the target time (whichever is sooner: measurement or log)
            t_target = min(next_meas_time, next_log_time)
            
            # Get ignition status
            ignition = self.sat.get_control_inputs(t_current)[1] > 0.5

            # Propagate to target simulation time
            self.kf.propagate(t_current, t_target, ignition, self.env, self.sat)
            t_current = t_target
            
            # Check for Measurement Event
            # Using a small epsilon for float comparison
            if abs(t_current - next_meas_time) < 1e-6:
                # Perform Measurement Update
                y_res, H, R = self._calculate_residual_and_jacobian(t_current,self.kf.x,meas_idx)

                if y_res is not None:
                    # --- MAHALANOBIS GATING CALCULATION ---
                    # Ensure y_res is a column vector for matrix math
                    y_vec = np.array(y_res).reshape(-1, 1)
                    
                    # Innovation Covariance (S)
                    S = H @ self.kf.P @ H.T + R
                    
                    # Squared Mahalanobis Distance: D^2 = y^T * S^-1 * y
                    # np.linalg.solve is numerically faster and more stable than explicit inversion
                    try:
                        S_inv_y = np.linalg.solve(S, y_vec)
                        d2 = (y_vec.T @ S_inv_y).item()
                        self.mahalanobis.append([t_current, len(y_vec), d2])
                    except np.linalg.LinAlgError:
                        print(f"Warning: S matrix is singular at t={t_current}")

                    self.kf.measurement_update(y_res, H, R)
                
                # Advance to next scheduled measurement
                meas_idx += 1
                
            # Check for Logging Event
            if abs(t_current - next_log_time) < 1e-6:
                self._log_state(log_idx)
                log_idx += 1
                if log_idx % 1000 == 0:
                    print(f'Logged {log_idx} steps.')

            # Exit simulation once we stop logging data 
            if log_idx >= self.n_log_steps:
                break
                
        print(f"Simulation Complete. Logged {log_idx} state estimations.")
        return self.est_states, self.est_covariances, self.mahalanobis

    def _log_state(self, idx: int):
        """
        Records the current EKF state and covariance matrix into the pre-allocated arrays.

        Parameters
        ----------
        idx : int
            The index in the `est_states` and `est_covariances` arrays where the 
            current state should be stored.
        """
        
        self.est_states[idx] = self.kf.x
        self.est_covariances[idx] = self.kf.P
