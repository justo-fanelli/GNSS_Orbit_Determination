import numpy as np
import pyshtools as pysh
from scipy.interpolate import interp1d, CubicHermiteSpline
from astropy import coordinates as coord
from astropy.time import Time
from astropy import units as u

OMEGA_E = 7.2921151467e-5   # Earth's angular velocity [rad/s]

class Environment:
    """
    Manages the physical simulation environment for the orbit determination kalman filter.

    This class handles the initialization and interpolation of the Earth's 
    Geopotential field (Spherical Harmonics), Third-Body ephemerides (Sun and Moon), 
    and the kinematic frame rotations between the inertial GCRF and Earth-fixed ITRF.

    Parameters
    ----------
    t_array : numpy.ndarray
        Array of simulation time epochs [s].
    dates_utc : list of datetime.datetime
        List of UTC datetime objects corresponding to the epochs in `t_array`.
    n_geopot : int, optional
        Maximum degree and order used for the Spherical Harmonics gravity model. 
        Default is 5.

    Attributes
    ----------
    t_array : numpy.ndarray
        The internal time array used for spline generation.
    n_geopot : int
        The truncated degree and order of the gravity model.
    Cnm : numpy.ndarray
        Fully normalized cosine coefficients for the geopotential.
    Snm : numpy.ndarray
        Fully normalized sine coefficients for the geopotential.
    rot_interps : list of scipy.interpolate.CubicHermiteSpline
        List of 9 splines representing the individual elements of the 3x3 
        GCRF-to-ITRF rotation matrix over time.
    sun_pos_interp : scipy.interpolate.interp1d
        Interpolator for the Sun's position vector in the GCRF frame [m].
    moon_pos_interp : scipy.interpolate.interp1d
        Interpolator for the Moon's position vector in the GCRF frame [m].
    """

    def __init__(self, t_array: np.ndarray, dates_utc: list, n_geopot: int = 5):
        self.t_array = t_array
        self.n_geopot = n_geopot
        
        # Load Spherical Harmonics coefficients (EGM2008)
        clm = pysh.datasets.Earth.EGM2008(lmax=n_geopot)
        clm_schmidt = clm.convert(normalization='unnorm')
        self.Cnm, self.Snm = np.array(clm_schmidt.coeffs)

        # Pre-calculate and spline GCRF/ITRF rotation matrices
        self._setup_rotation_matrices(dates_utc, len(t_array))

        # Retrieve and spline Sun/Moon coordinates in GCRF
        sun_coord = coord.get_sun(Time(dates_utc, format='datetime'))
        moon_coord = coord.get_body('moon', Time(dates_utc, format='datetime'))
        
        self.sun_pos_interp = interp1d(t_array, sun_coord.cartesian.xyz.to(u.m).value.T, 
                                       axis=0, bounds_error=False)
        self.moon_pos_interp = interp1d(t_array, moon_coord.cartesian.xyz.to(u.m).value.T, 
                                        axis=0, bounds_error=False)

    def _setup_rotation_matrices(self, dates: list, steps: int):
        """
        Precomputes and splines the GCRF to ITRF rotation matrices.
        This bypasses the massive computational overhead of calling Astropy 
        coordinate transformations inside the RK45 integration loop.
        """
        basis_vectors = np.tile(np.eye(3), (steps, 1))
        pos_repr = coord.CartesianRepresentation(basis_vectors[:, 0], basis_vectors[:, 1], 
                                                 basis_vectors[:, 2], unit=u.m)
        time_obj = np.repeat(Time(dates, scale='utc'), 3)
        
        gcrf = coord.GCRS(pos_repr, obstime=time_obj)
        itrf = gcrf.transform_to(coord.ITRS(obstime=time_obj))
        
        rot_vecs = itrf.cartesian.xyz.to(u.m).value
        rot_matrices = np.transpose(np.reshape(rot_vecs, (3, 3, steps), order='F'), axes=(2, 0, 1))
        
        # Calculate derivatives for Cubic Hermite Spline using Earth's angular velocity
        S_omega = np.array([[0, OMEGA_E, 0], [-OMEGA_E, 0, 0], [0, 0, 0]])
        d_rot = np.matmul(S_omega, rot_matrices)
        
        self.rot_interps = []
        for i in range(3):
            for j in range(3):
                self.rot_interps.append(CubicHermiteSpline(self.t_array, rot_matrices[:, i, j], d_rot[:, i, j]))

    def get_rotation_matrix(self, t: float) -> np.ndarray:
        """
        Evaluates the splined rotation matrix at a specific simulation time.

        Parameters
        ----------
        t : float
            Current simulation time [s].

        Returns
        -------
        numpy.ndarray
            The 3x3 direction cosine matrix transforming vectors from the 
            GCRF (inertial) frame to the ITRF (Earth-fixed) frame.
        """
        R = np.zeros((3, 3))
        for k, func in enumerate(self.rot_interps):
            i, j = divmod(k, 3)
            R[i, j] = func(t)
        return R

    def get_third_bodies(self, t: float) -> tuple:
        """
        Evaluates the position of the Sun and Moon at a specific simulation time.

        Parameters
        ----------
        t : float
            Current simulation time [s].

        Returns
        -------
        tuple
            - r_sun (numpy.ndarray): Position vector of the Sun in GCRF [m]. Shape: (3,).
            - r_moon (numpy.ndarray): Position vector of the Moon in GCRF [m]. Shape: (3,).
        """
        return self.sun_pos_interp(t), self.moon_pos_interp(t)
