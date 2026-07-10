from pathlib import Path
import numpy as np

C_LIGHT = 299792458.0   # Speed of light [m/s]

class NavSatSystems:
    """
    Manages GNSS constellations and measurement simulation.

    This class stores constellation-specific physical constants (frequencies, 
    chip lengths, SISRE) and generates simulated noisy observations (pseudorange 
    and pseudorange-rate) based on the receiver's truth state and theoretical 
    tracking loop thermal noise models (DLL/FLL).

    Parameters
    ----------
    gnss_names : list of str
        List of strings representing the constellations to simulate 
        (e.g., ['GPS', 'Galileo', 'GLONASS', 'BDS']).

    Attributes
    ----------
    constellations : list of str
        The active GNSS constellations being simulated.
    sisre : dict
        Signal-in-Space Range Error bounds for each constellation [m].
    freqs : dict
        L1/E1/G1 carrier frequencies for each constellation [Hz].
    chip_length : dict
        Code chip lengths for the PRN sequences of each constellation [m].
    raw_data : dict
        Nested dictionary holding the raw ephemeris and visibility arrays 
        loaded from disk for each constellation.
    aligned_data : dict or None
        Nested dictionary holding GNSS states pre-interpolated to the exact 
        measurement epochs to optimize runtime execution.
    """

    def __init__(self, gnss_names: list):
        self.constellations = gnss_names
        self.sisre = {'GPS': 0.8, 'Galileo': 1.1, 'GLONASS': 1.4, 'BDS': 1.0}
        self.freqs = {'GPS': 1575.42e6, 'Galileo': 1575.42e6, 'GLONASS': 1602e6, 'BDS': 1575.42e6}
        self.chip_length = {'GPS': 293.05, 'Galileo': 293.05, 'GLONASS': 586.68, 'BDS': 146.52}
        self.raw_data = {name: {} for name in gnss_names}
        self.aligned_data = None

    def load_data(self, path_pos: Path, path_vis: Path):
        """
        Loads processed GNSS ephemeris and visibility data into memory.

        Parameters
        ----------
        path_pos : pathlib.Path
            Directory path to the `.npz` ephemeris files.
        path_vis : pathlib.Path
            Directory path to the `.dat` visibility and C/N0 files.
        """
        for const_name in self.constellations: 

            if const_name == 'BDS':
                # Load both .npz files
                gnss_data_meo = np.load(path_pos / 'BDS_MEO.npz')
                gnss_data_geo = np.load(path_pos / 'BDS_GEO_IGSO.npz')
                
                # Concatenate the specific arrays along the satellite axis (axis 0)
                raw_positions = np.concatenate((gnss_data_meo['positions'], gnss_data_geo['positions']), axis=0)
                raw_velocities = np.concatenate((gnss_data_meo['velocities'], gnss_data_geo['velocities']), axis=0)
                
                # Time arrays are identical, so we just extract them from one
                jd = gnss_data_meo['jd']
                fr = gnss_data_meo['fr']
                
            else:
                # Load standard constellations (GPS, Galileo, GLONASS)
                gnss_data = np.load(path_pos / f'{const_name}.npz')
                
                raw_positions = gnss_data['positions']
                raw_velocities = gnss_data['velocities']
                jd = gnss_data['jd']
                fr = gnss_data['fr']
            
            # Extract and format the ephemeris data
            # Transpose from (Sats, Timesteps, 3) to (Timesteps, Sats, 3)
            # Multiply by 1e3 to convert from km to meters for the EKF
            self.raw_data[const_name]['pos'] = raw_positions.transpose(1, 0, 2) * 1e3
            self.raw_data[const_name]['vel'] = raw_velocities.transpose(1, 0, 2) * 1e3
            
            # Calculate elapsed time in seconds from the initial epoch
            jd = gnss_data['jd']
            fr = gnss_data['fr']
            elapsed_days = (jd + fr) - (jd[0] + fr[0])
            self.raw_data[const_name]['times_pos'] = elapsed_days * 86400.0

            # Load visibility and C/N0 data
            # Position/velocity and visibility data don't necessarily have the same timebase
            vis_data = np.loadtxt(path_vis/const_name/'access_data.dat',skiprows=2)
            self.raw_data[const_name]['vis'] = vis_data[:,1:]
            self.raw_data[const_name]['times_vis'] = vis_data[:,0]
            ctnr_data = np.loadtxt(path_vis/const_name/'ctnr_unfiltered.dat',skiprows=2)
            self.raw_data[const_name]['ctnr'] = ctnr_data[:,1:]


    def prealign_constellations(self, t_meas_array: np.ndarray, eor_time_offset_s: float):
        """
        Pre-interpolates GNSS data to the measurement epochs.

        Parameters
        ----------
        t_meas_array : numpy.ndarray
            Array of scheduled measurement epochs [s].
        eor_time_offset_s : float
            Time offset between the GNSS data start epoch and the receiver trajectory 
            start epoch [s].
        """
        
        self.aligned_data = {name: {} for name in self.constellations}

        # Shift the measurement array to match the GNSS ephemeris timebase
        shifted_t_meas = t_meas_array + eor_time_offset_s
        
        for const in self.constellations:
            # Extract raw data
            raw_t = self.raw_data[const]['times_pos']
            raw_t_vis = self.raw_data[const]['times_vis']
            raw_pos = self.raw_data[const]['pos'] # (T_raw, Sats, 3)
            raw_vel = self.raw_data[const]['vel']
            raw_vis = self.raw_data[const]['vis']
            raw_ctnr = self.raw_data[const]['ctnr']
            
            n_sats = raw_pos.shape[1]
            n_steps = len(t_meas_array)
            
            # Initialize containers for this constellation
            aligned_pos = np.zeros((n_steps, n_sats, 3))
            aligned_vel = np.zeros((n_steps, n_sats, 3))
            aligned_vis = np.zeros((n_steps, n_sats), dtype=int)
            aligned_ctnr = np.zeros((n_steps, n_sats))
            
            # Interpolate every satellite
            # We assume raw_t is sorted. np.interp is fast.
            for s in range(n_sats):
                for axis in range(3):
                    aligned_pos[:, s, axis] = np.interp(shifted_t_meas, raw_t, raw_pos[:, s, axis])
                    aligned_vel[:, s, axis] = np.interp(shifted_t_meas, raw_t, raw_vel[:, s, axis])
                
                # Visibility: Nearest Neighbor Interpolation
                # We interpret > 0.5 as "visible"
                vis_interp = np.interp(shifted_t_meas, raw_t_vis, raw_vis[:, s])
                aligned_vis[:, s] = (vis_interp > 0.5).astype(int)
                aligned_ctnr[:, s] = np.interp(shifted_t_meas, raw_t_vis, raw_ctnr[:, s])

            # Store
            self.aligned_data[const]['pos'] = aligned_pos
            self.aligned_data[const]['vel'] = aligned_vel
            self.aligned_data[const]['vis'] = aligned_vis
            self.aligned_data[const]['ctnr'] = aligned_ctnr

    def sigma_pr_dll(self, const: str, cn0_dbhz: float, T: float = 0.02, 
                     B: float = 1.0, D: float = 0.5) -> float:
        """
        Calculates the Delay Lock Loop (DLL) thermal noise contribution 
        to the pseudorange standard deviation.

        Parameters
        ----------
        const : str
            Name of the GNSS constellation.
        cn0_dbhz : float
            Carrier-to-Noise ratio of the received signal [dB-Hz].
        T : float, optional
            Predetection integration time [s]. Default is 0.02 s (20 ms).
        B : float, optional
            Code loop noise bandwidth [Hz]. Default is 1.0 Hz.
        D : float, optional
            Early-to-late correlator spacing [chips]. Default is 0.5 chips.

        Returns
        -------
        float
            Standard deviation of the pseudorange error [m].
        """

        cn0_lin = 10**(cn0_dbhz / 10.0)
        return self.chip_length[const] * np.sqrt(B * D / (2 * cn0_lin) * (1 + 2 / ((2 - D) * T * cn0_lin)))
    
    def sigma_rate_fll(self, const: str, cn0_dbhz: float, T: float = 0.005, 
                       B: float = 10.0) -> float:
        """
        Calculates the Frequency Lock Loop (FLL) thermal noise contribution 
        to the pseudorange-rate standard deviation.

        Parameters
        ----------
        const : str
            Name of the GNSS constellation.
        cn0_dbhz : float
            Carrier-to-Noise ratio of the received signal [dB-Hz].
        T : float, optional
            Predetection integration time [s]. Default is 0.005 s (5 ms).
        B : float, optional
            Frequency loop noise bandwidth [Hz]. Default is 10.0 Hz.

        Returns
        -------
        float
            Standard deviation of the pseudorange-rate error [m/s].
        """

        cn0_lin = 10**(cn0_dbhz / 10.0)
        wl = C_LIGHT / self.freqs[const]
        return wl / (2 * np.pi * T) * np.sqrt(4 * B / cn0_lin * (1 + 1 / (T * cn0_lin)))

    def compute_gnss_observables(self, t_idx: int, rx_state: np.ndarray) -> tuple:
        """
        Calculates theoretical measurements h(x) and the Jacobian matrix H 
        evaluated at the current estimated receiver state.

        Parameters
        ----------
        t_idx : int
            The index corresponding to the current measurement time epoch.
        rx_state : numpy.ndarray
            The current estimated 8-state vector of the receiver: 
            [x, y, z, vx, vy, vz, clock_bias, clock_drift]. Shape: (8,).

        Returns
        -------
        tuple
            - z_pred (numpy.ndarray): Predicted pseudorange and pseudorange-rate 
              measurements. Shape: (2 * N_visible,).
            - H (numpy.ndarray): Observation Jacobian matrix evaluated at rx_state. 
              Shape: (2 * N_visible, 8).
            - R (numpy.ndarray): Measurement noise covariance matrix based on C/N0. 
              Shape: (2 * N_visible, 2 * N_visible).
            Returns (None, None, None) if no satellites are visible.
        """

        rx_pos = rx_state[0:3]
        rx_vel = rx_state[3:6]
        rx_clk_bias = rx_state[6]
        rx_clk_drift = rx_state[7]
    
        z_pr, z_rate, R_pr, R_rate, H_pr, H_rate = [], [], [], [], [], []
        
        for const in self.constellations:
            # 1. Retrieve Data
            sats_pos = self.aligned_data[const]['pos'][t_idx]
            sats_vel = self.aligned_data[const]['vel'][t_idx]
            sats_vis = self.aligned_data[const]['vis'][t_idx]
            sats_ctnr = self.aligned_data[const]['ctnr'][t_idx]

            # Check visibility at the nearest epoch
            vis_indices = np.where(sats_vis > 0)[0]
            n_vis = len(vis_indices)
            
            if n_vis == 0:
                continue
            
            pos_vis_sats = sats_pos[vis_indices]
            vel_vis_sats = sats_vel[vis_indices]
            ctnr_vis_sats = sats_ctnr[vis_indices]
                
            # Geometry
            rel_pos = pos_vis_sats - rx_pos
            dist = np.linalg.norm(rel_pos,axis=1)
            unit_vecs = rel_pos / dist[:, None]
            rel_vel = vel_vis_sats - rx_vel
            range_rate = np.sum(rel_vel * unit_vecs, axis=1)

            # Modeled PR = Geom Range + Clock Bias
            pr = dist + rx_clk_bias
            z_pr.append(pr)
    
            # Modeled Doppler = Range Rate + Clock Drift
            rate = range_rate + rx_clk_drift 
            z_rate.append(rate)

            # Jacobian blocks for measurement matrix H
            # PR block
            H_pr_chunk = np.zeros((n_vis, 8))
            H_pr_chunk[:, 0:3] = -unit_vecs
            H_pr_chunk[:, 6] = 1.0
            H_pr.append(H_pr_chunk)
                
            # Doppler (range-rate) block
            H_rate_chunk = np.zeros((n_vis,8))
            H_rate_chunk[:,3:6] = -unit_vecs
            H_rate_chunk[:,7] = 1.0
            H_rate.append(H_rate_chunk)
                
            # Noise covariance matrix (for measurement simulation)
            sigma_pr = [self.sigma_pr_dll(const,ctnr)**2 + self.sisre[const] for ctnr in ctnr_vis_sats]
            sigma_rate = [self.sigma_rate_fll(const,ctnr)**2 for ctnr in ctnr_vis_sats]
            R_pr.extend(sigma_pr)
            R_rate.extend(sigma_rate)

        # Return None if no satellites visible
        if not z_pr:
            return None, None, None

        # Assemble Matrices
        z = np.concatenate([np.concatenate(z_pr), np.concatenate(z_rate)])
        H = np.vstack(H_pr + H_rate)
        R = np.diag(R_pr + R_rate)
        
        return z, H, R
    
    def get_measurement(self, t_idx, rx_true_state):
        """
        Simulates noisy sensor measurements using the truth state.

        Takes the exact theoretical observables and injects Gaussian white noise 
        scaled by the environment's C/N0-derived covariance matrix.

        Parameters
        ----------
        t_idx : int
            The index corresponding to the current measurement time epoch.
        rx_true_state : numpy.ndarray
            The interpolated true 8-state vector of the receiver.

        Returns
        -------
        tuple
            - z_meas (numpy.ndarray): Noisy pseudorange and pseudorange-rate 
              measurements. Shape: (2 * N_visible,).
            - R (numpy.ndarray): The diagonal measurement noise covariance matrix 
              associated with these readings. Shape: (2 * N_visible, 2 * N_visible).
            Returns (None, None) if no satellites are visible.
        """

        z_clean, _, R = self.compute_gnss_observables(t_idx, rx_true_state)
        
        if z_clean is None:
            return None, None
            
        # Add Noise consistent with R
        noise = np.random.normal(0, np.sqrt(np.diag(R)))
        z_meas = z_clean + noise
        
        return z_meas, R
