"""
Module containing the Extended Kalman Filter (EKF) implementation, environment 
models, and simulation orchestration for GNSS-based Orbit Determination.
"""

import numpy as np
from scipy.integrate import RK45

# Import our custom geopotential module
from orbit_determination.geopotential import gravity_sh_acel
from orbit_determination.ekf_environment import Environment
from orbit_determination.ekf_spacecraft import Spacecraft

# --- Physical Constants (SI Units) ---
C_LIGHT = 299792458.0              # Speed of light [m/s]
MU_E = 3.986004415e14              # Earth's gravitational parameter [m^3/s^2]
MU_M = 4.902800118e12              # Moon's gravitational parameter [m^3/s^2]
MU_S = 1.32712440018e20            # Sun's gravitational parameter [m^3/s^2]
R_EARTH = 6378.1363e3              # Earth's average equatorial radius [m]
OMEGA_E = 7.2921151467e-5          # Earth's angular velocity [rad/s]
SUN_RAD_PRESSURE = 4.56e-6         # Solar radiation pressure at 1 A.U. [N/m^2]


class OrbitalEKF:
    """
    Extended Kalman Filter (EKF) for spacecraft Orbit Determination.

    This class maintains the 8-element state vector and its corresponding 
    error covariance matrix. It handles the continuous-time propagation 
    (Time Update) using an RK45 integrator and discrete-time measurement 
    assimilation (Measurement Update) from GNSS observables.

    State Vector (8x1): 
    [X, Y, Z, Vx, Vy, Vz, ClockBias, ClockDrift]

    Parameters
    ----------
    x0 : numpy.ndarray
        Initial state vector estimate. Shape: (8,).
    P0 : numpy.ndarray
        Initial state error covariance matrix. Shape: (8, 8).
    noise_params : dict
        Dictionary containing spectral densities for process noise generation. 
        Must include 'Sa' (acceleration noise), 'Sf' (clock phase noise), 
        and 'Sg' (clock frequency noise).

    Attributes
    ----------
    x : numpy.ndarray
        Current estimated state vector. Shape: (8,).
    P : numpy.ndarray
        Current estimated state error covariance matrix. Shape: (8, 8).
    noise : dict
        Stored noise spectral densities.
    ignition : bool
        Current status of the spacecraft's thruster (True if firing, False if coasting).
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray, noise_params: dict):
        self.x = x0
        self.P = P0
        self.noise = noise_params 
        self.ignition = False

    def get_process_noise(self, dt: float) -> np.ndarray:
        """
        Constructs the discrete-time Process Noise Covariance matrix (Q).

        Parameters
        ----------
        dt : float
            Time step duration [s].

        Returns
        -------
        numpy.ndarray
            Process noise covariance matrix. Shape: (8, 8).
        """
        Q = np.zeros((8,8))
        # Position and velocity
        Qxx = self.noise['Sa']*dt**3/3
        Qvv = self.noise['Sa']*dt 
        Qxv = self.noise['Sa']*dt**2/2
        # Clock
        Qbb = (self.noise['Sf']*dt + self.noise['Sg']*dt**3 / 3)*C_LIGHT**2
        Qdd = self.noise['Sg']*dt*C_LIGHT**2
        Qbd = self.noise['Sg']*dt**2 / 2 * C_LIGHT**2
        
        # Position and velocity block
        # We assume that x,y,z coordinates share the same elements 
        Q[0:3, 0:3] = np.eye(3) * Qxx
        Q[3:6, 3:6] = np.eye(3) * Qvv 
        Q[0:3, 3:6] = np.eye(3) * Qxv
        Q[3:6, 0:3] = np.eye(3) * Qxv
    
        # Clock block
        Q[6,6] = Qbb; Q[7,7] = Qdd
        Q[6,7] = Qbd; Q[7,6] = Qbd

        return Q 
    
    def _sat_sun_los(self, r_rcver: np.ndarray, r_sun: np.ndarray) -> bool:
        """Determines if the Sun is eclipsed by the Earth."""
        rel_pos = r_sun - r_rcver
        k = -np.dot(r_rcver, rel_pos)/(np.linalg.norm(rel_pos))**2
        if (k >= 1 or k <= 0):
            return True
        if (np.linalg.norm(r_rcver + k*rel_pos) > R_EARTH):
            # earth doesn't get in between the sun and moon
            return True
        
        return False

    def _deriv_func(self, t: float, state: np.ndarray, env: Environment, sat: Spacecraft) -> np.ndarray:
        """
        Continuous-time dynamics equation f(x, u, t) evaluated by the RK45 integrator.

        Computes the total acceleration acting on the spacecraft, including 
        geopotential, third-body (Sun/Moon), solar radiation pressure, and thrust.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        state : numpy.ndarray
            Current integrated state vector. Shape: (8,).
        env : Environment
            Instance of the environment manager for physical calculations.
        sat : Spacecraft
            Instance of the spacecraft containing mass and area properties.

        Returns
        -------
        numpy.ndarray
            State derivative vector [vx, vy, vz, ax, ay, az, drift, 0]. Shape: (8,).
        """
        # GCRF (ECI) position and velocity
        pos = state[0:3]
        vel = state[3:6]
        drift = state[7]
        
        # Transform ECI position to ECEF
        R_mat = env.get_rotation_matrix(t)
        pos_ecef = R_mat @ pos
        
        # Calculate geopotential acceleration
        acc_geo_ecef = gravity_sh_acel(pos_ecef, env.Cnm, env.Snm, MU_E, R_EARTH)
        # Rotate back to ECI frame
        acc_geo_eci = R_mat.T @ acc_geo_ecef
        
        # Calculate third body acceleration due to the sun and moon
        r_sun, r_moon = env.get_third_bodies(t)
        acc_sun = MU_S * ( (r_sun - pos)/np.linalg.norm(r_sun - pos)**3 - r_sun/np.linalg.norm(r_sun)**3 )
        acc_moon = MU_M * ( (r_moon - pos)/np.linalg.norm(r_moon - pos)**3 - r_moon/np.linalg.norm(r_moon)**3 )
        
        # Acceleration due to thrust
        mass, _, thrust_dir = sat.get_control_inputs(t)
        acc_thrust = (sat.fthr / mass * thrust_dir) if self.ignition else np.zeros(3)

        # Solar radiation pressure contribution
        acc_srp = ((sat.Cr*SUN_RAD_PRESSURE*sat.area/mass) * (pos-r_sun)/np.linalg.norm(pos-r_sun)) if \
            self._sat_sun_los(pos, r_sun) else np.zeros(3)
        
        acc_total = acc_geo_eci + acc_sun + acc_moon + acc_thrust + acc_srp
        
        return np.concatenate([vel, acc_total, [drift, 0]])

    def propagate(self, t0: float, tf: float, ignition: bool, env: Environment, sat: Spacecraft):
        """
        Time Update Step: Propagates the state estimate and covariance matrix forward in time.

        Uses an RK45 numerical integrator for the non-linear state dynamics and 
        a Central Gravity approximation for the State Transition Matrix (Phi).

        Parameters
        ----------
        t0 : float
            Initial propagation time [s].
        tf : float
            Target propagation time (next measurement epoch) [s].
        ignition : bool
            Flag indicating whether the spacecraft's thruster is currently firing.
        env : Environment
            Instance of the environment manager.
        sat : Spacecraft
            Instance of the spacecraft.
        """
        self.ignition = ignition
        # Integrate State 
        solver = RK45(lambda t, y: self._deriv_func(t, y, env, sat), t0=t0, y0=self.x, t_bound=tf,
                rtol=1e-11,atol=[1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-3, 1e-6])
        while solver.status == 'running':
            solver.step()

        # Central gravity force due to the earth is good enough to propagate
        # the error-variance matrix P
        r_vec = self.x[:3]
        r_norm = np.linalg.norm(r_vec)
        # The elements of F are calculated as the jacobian of the derivative function f(x)
        # that defines the time evolution of the system
        F = np.zeros((8,8))
        F[0:3, 3:6] = np.eye(3) # Identity block for kinematics
        F[6, 7] = 1.0
        F[3:6, 0:3] = -(MU_E/r_norm**3)*np.eye(3) + 3*(MU_E/r_norm**5)*np.outer(r_vec, r_vec)
            
        # Calculate the state transition matrix, approximating expm(F*dt) as I + F*dt
        dt = tf - t0
        Phi = np.eye(8) + F * dt 
        Q = self.get_process_noise(dt)
        self.P = Phi @ self.P @ Phi.T + Q

        # Update state
        self.x = solver.y
        

    def measurement_update(self, v: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Measurement Update Step: Assimilates sensor data to correct the state estimate.

        Calculates the Kalman Gain and updates the state vector and covariance 
        matrix using the innovation (residual).

        Parameters
        ----------
        v : numpy.ndarray
            Measurement innovation vector (z_measured - z_predicted). Shape: (M,).
        H : numpy.ndarray
            Observation Jacobian matrix. Shape: (M, 8).
        R : numpy.ndarray
            Measurement noise covariance matrix. Shape: (M, M).
        """
        # Covariance matrix of residuals Qvv
        Qvv = H @ self.P @ H.T + R
        # Kalman gain K
        K = self.P @ H.T @ np.linalg.inv(Qvv)
        self.x = self.x + K @ v 
        self.P = (np.eye(8) - K @ H) @ self.P


