import numpy as np
from scipy.interpolate import CubicHermiteSpline

class Spacecraft:
    """
    This class stores the spacecraft's physical properties (mass, area, engine 
    specifications) and utilizes Cubic Hermite Splines to continuously interpolate 
    the "True" reference trajectory and clock states. This truth state is then 
    used to generate the GNSS measurements.

    Parameters
    ----------
    t_data_array : numpy.ndarray
        Array of time epochs corresponding to the truth data [s].
    true_pos : numpy.ndarray
        True position vectors in the GCRF frame [m]. Shape: (N, 3).
    true_vel : numpy.ndarray
        True velocity vectors in the GCRF frame [m/s]. Shape: (N, 3).
    true_clk_bias : numpy.ndarray
        True receiver clock bias [m]. Shape: (N,).
    true_clk_drift : numpy.ndarray
        True receiver clock drift [m/s]. Shape: (N,).
    mass_func : callable
        Function that takes time `t` and returns the spacecraft's mass [kg].
    ignition_func : callable
        Function that takes time `t` and returns the thruster ignition status 
        (e.g., 1.0 for firing, 0.0 for coasting).
    fthr_dir_func : callable
        Function that takes time `t` and returns the 3D thrust direction unit vector.
    mass_props : dict
        Dictionary containing constant physical properties of the spacecraft. 
        Must include keys: 'mass' [kg], 'Isp' [s], 'thrust' [N], 'area' [m^2], 
        and 'Cr' (radiation pressure coefficient).

    Attributes
    ----------
    init_mass : float
        Initial wet mass of the spacecraft [kg].
    Isp : float
        Specific impulse of the primary propulsion system [s].
    fthr : float
        Magnitude of the engine's thrust [N].
    area : float
        Cross-sectional area exposed to solar radiation [m^2].
    Cr : float
        Solar radiation pressure coefficient (typically between 1.0 and 2.0).
    pos_spline : scipy.interpolate.CubicHermiteSpline
        Continuous interpolator for the true orbital position and velocity.
    clk_spline : scipy.interpolate.CubicHermiteSpline
        Continuous interpolator for the true clock bias and drift.
    mass_func : callable
        Stored function for mass evaluation.
    ignition_func : callable
        Stored function for engine ignition status.
    fthr_dir_func : callable
        Stored function for thrust pointing vector.
    """

    def __init__(self, t_data_array, true_pos, true_vel, true_clk_bias, true_clk_drift, 
                 mass_func, ignition_func, fthr_dir_func, mass_props):
        self.init_mass = mass_props['mass']
        self.Isp = mass_props['Isp']
        self.fthr = mass_props['thrust']
        self.area = mass_props['area']
        self.Cr = mass_props['Cr']

        self.pos_spline = CubicHermiteSpline(t_data_array, true_pos, true_vel, axis=0)
        self.clk_spline = CubicHermiteSpline(t_data_array, true_clk_bias, true_clk_drift, axis=0)
        
        self.mass_func = mass_func
        self.ignition_func = ignition_func
        self.fthr_dir_func = fthr_dir_func
    
    def get_true_state(self, t: float) -> np.ndarray:
        """
        Evaluates the interpolated true state vector of the spacecraft at a given time.

        Parameters
        ----------
        t : float
            Simulation time [s].

        Returns
        -------
        numpy.ndarray
            The 8-state truth vector: 
            [x, y, z, vx, vy, vz, clock_bias, clock_drift]. Shape: (8,).
        """
        # The derivative of the position spline yields the velocity
        # The derivative of the bias spline yields the drift
        return np.concatenate([self.pos_spline(t), self.pos_spline.derivative()(t), 
                               [self.clk_spline(t)], [self.clk_spline.derivative()(t)]])

    def get_control_inputs(self, t: float) -> tuple:
        """
        Evaluates the spacecraft's physical properties and thruster state at a given time.

        Parameters
        ----------
        t : float
            Simulation time [s].

        Returns
        -------
        tuple
            Contains:
            - mass (float): Current spacecraft mass [kg].
            - ignition (float): Current engine firing status (1.0 = on, 0.0 = off).
            - thrust_dir (numpy.ndarray): Current 3D thrust unit vector. Shape: (3,).
        """
        return self.mass_func(t), self.ignition_func(t), self.fthr_dir_func(t)
