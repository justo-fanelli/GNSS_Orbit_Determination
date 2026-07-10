import json
import numpy as np
from pathlib import Path
from sgp4.api import Satrec, SatrecArray, jday
from astropy.time import Time
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
import astropy.units as u
from datetime import datetime

class ConstellationSimulator:
    """
    Simulates GNSS constellation orbital dynamics using the SGP4 analytical model.

    This class loads Orbital Mean Elements (OMMs), 
    filters non-operational satellites, and performs vectorized propagation. 
    It outputs states in the GCRF frame.

    Attributes
    ----------
    data_path : Path
        Base directory path for GNSS data and ephemeris exports.
    excluded_sats : set
        A set of NORAD catalog IDs representing non-operational satellites.
    """

    def __init__(self, data_path: Path):
        """
        Initializes the simulator and loads the exclusion list for inactive satellites.

        Parameters
        ----------
        data_path : Path
            The root directory containing the OMM data and the inactive satellites.
        """
        self.data_path = Path(data_path)
        self.excluded_sats = self._load_excluded_sats()

    def _load_excluded_sats(self):
        """
        Loads the dictionary of non-operational satellites to prevent them 
        from being propagated.

        Returns
        -------
        set
            Flattened set of inactive NORAD IDs for O(1) lookup time.
        """
        excluded_file = self.data_path / 'non_operational_sats.txt'
        try:
            with open(excluded_file, 'r') as f:
                no_op_dict = json.load(f)
            # Flatten the IDs into a set    
            return set(sat_id for lst in no_op_dict.values() for sat_id in lst)
        except FileNotFoundError:
            print(f"Warning: Exclusion file not found at {excluded_file}. Propagating all.")
            return set()

    def propagate_constellation(self, omm_file: Path, jd_start: float, fr_start: float, 
                                duration_s: float, dt: float):
        """
        Loads orbital elements for a single constellation and propagates them vectorially.

        Parameters
        ----------
        omm_file : Path
            Path to the JSON file containing OMM data.
        jd_start : float
            Initial Julian Date (integer part).
        fr_start : float
            Initial Julian Date (fractional part).
        duration_s : float
            Total simulation duration in seconds.
        dt : float
            Time step for the propagation in seconds.

        Returns
        -------
        tuple
            Contains:
            - sat_names (list): Names of successfully propagated satellites.
            - positions_gcrf (numpy.ndarray): Position vectors in GCRF frame (Sats, Steps, 3) [km].
            - velocities_gcrf (numpy.ndarray): Velocity vectors in GCRF frame (Sats, Steps, 3) [km/s].
            - error_codes (numpy.ndarray): SGP4 internal error codes.
            - jd_array (numpy.ndarray): Array of Julian Date integers.
            - fr_array (numpy.ndarray): Array of Julian Date fractions.
        """
        satrec_list = []
        sat_names = []
        
        with open(omm_file, 'r') as f:
            omm_data = json.load(f)
            for sat in omm_data:
                norad_id_str = sat.get('NORAD_CAT_ID') # Safely get the NORAD ID
                if not norad_id_str:
                    continue
                    
                norad_id = int(norad_id_str)
                
                # Filter out if it's in the non-operational list
                if norad_id not in self.excluded_sats:
                    line1 = sat.get('TLE_LINE1')
                    line2 = sat.get('TLE_LINE2')
                    
                    fallback_name = sat.get('TLE_LINE0', f'UNKNOWN_{norad_id}').replace('0 ', '').strip() # Fallback naming
                    sat_name = sat.get('OBJECT_NAME', fallback_name)
                    
                    if line1 and line2:
                        record = Satrec.twoline2rv(line1, line2)
                        satrec_list.append(record)
                        sat_names.append(sat_name)

        if not satrec_list:
            print(f"Warning: No valid satellites found in {omm_file}")
            return [], np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        # Pack into SatrecArray to enable C++ vectorization
        sat_array = SatrecArray(satrec_list)

        steps = int(duration_s / dt)
        dt_days = dt / 86400.0
        
        jd_array = np.full(steps, jd_start)
        fr_array = fr_start + np.arange(steps) * dt_days

        error_codes, positions, velocities = sat_array.sgp4(jd_array, fr_array)
        
        # Transform to GCRF frame
        pos_t = positions.transpose(2, 0, 1)
        vel_t = velocities.transpose(2, 0, 1)
        
        teme_p = CartesianRepresentation(x=pos_t[0]*u.km, y=pos_t[1]*u.km, z=pos_t[2]*u.km)
        teme_v = CartesianDifferential(d_x=vel_t[0]*u.km/u.s, d_y=vel_t[1]*u.km/u.s, d_z=vel_t[2]*u.km/u.s)
        
        epochs = Time(jd_array[np.newaxis, :], fr_array[np.newaxis, :], format='jd', scale='utc')
        
        teme_frame = TEME(teme_p.with_differentials(teme_v), obstime=epochs)
        gcrf_frame = teme_frame.transform_to(GCRS(obstime=epochs))
        
        positions_gcrf = gcrf_frame.cartesian.xyz.value.transpose(1, 2, 0)
        velocities_gcrf = gcrf_frame.velocity.d_xyz.value.transpose(1, 2, 0)

        return sat_names, positions_gcrf, velocities_gcrf, error_codes, jd_array, fr_array

    def run_all_constellations(self, dt_initial: datetime, sim_duration_s: float, sim_step_s: float,
                               export_dir: Path):
        """
        Executes the propagation pipeline for all configured GNSS constellations 
        and exports the ephemeris data to compressed NumPy files with extension .npz.

        Parameters
        ----------
        dt_initial : datetime
            The initial UTC epoch for the simulation.
        sim_duration_s : float
            Total duration of the simulation in seconds.
        sim_step_s : float
            Time step for the simulation in seconds.
        """
        constellations = {
            'GPS': self.data_path / 'GPS.txt',
            'Galileo': self.data_path / 'Galileo.txt',
            'GLONASS': self.data_path / 'GLONASS.txt',
            'BDS': self.data_path / 'BDS.txt'
        }

        jd_initial, fr_initial = jday(dt_initial.year, dt_initial.month, dt_initial.day, 
                                      dt_initial.hour, dt_initial.minute, 
                                      dt_initial.second + 1e-6*dt_initial.microsecond)

        # Loop through each constellation and process them individually
        for const_name, file_name in constellations.items():
            print(f"Processing {const_name} constellation...")
                
            names, r, v, err, jd, fr = self.propagate_constellation(
                omm_file=file_name, jd_start=jd_initial, fr_start=fr_initial,
                duration_s=sim_duration_s, dt=sim_step_s
            )
            
            if not names:
                continue
            names_arr = np.array(names)
            
            export_dir.mkdir(parents=True, exist_ok=True)

            # Discriminate BeiDou MEO vs GEO/IGSO based on orbit radius
            if const_name == 'BDS':
                initial_radii = np.linalg.norm(r[:, 0, :], axis=1)
                meo_mask = initial_radii < 35000.0 # MEOs are < 35,000 km
                
                # Export MEO
                out_meo = export_dir / 'BDS_MEO.npz'
                np.savez_compressed(out_meo, names=names_arr[meo_mask], positions=r[meo_mask], 
                                    velocities=v[meo_mask], errors=err[meo_mask], jd=jd, fr=fr)
                print(f" -> Exported {np.sum(meo_mask)} active MEO satellites to '{out_meo}'")

                # Export GEO/IGSO
                out_geo_igso = export_dir / 'BDS_GEO_IGSO.npz'
                np.savez_compressed(out_geo_igso, names=names_arr[~meo_mask], positions=r[~meo_mask], 
                                    velocities=v[~meo_mask], errors=err[~meo_mask], jd=jd, fr=fr)
                print(f" -> Exported {np.sum(~meo_mask)} active GEO/IGSO satellites to '{out_geo_igso}'\n")
                
            else:
                out_filename = export_dir / f'{const_name}.npz'
                np.savez_compressed(out_filename, names=names_arr, positions=r, 
                                    velocities=v, errors=err, jd=jd, fr=fr)
                print(f" -> Exported {len(names)} active satellites to '{out_filename}'\n")

        print("All constellations successfully processed and exported.")
