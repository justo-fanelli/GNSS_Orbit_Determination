"""
Module for simulating and exporting GNSS constellation visibility data.
"""

import numpy as np
from pathlib import Path
from orbit_determination.access import sat_visibility
from orbit_determination.time_utils import julian_to_datetime

class VisibilitySimulator:
    """
    Simulates and exports GNSS visibility data for a given receiver trajectory.

    Attributes
    ----------
    data_path : Path
        Root directory for the project data.
    rcver_data_path : Path
        Path to the receiver trajectory and attitude file.
    rcver_antenna_dir_body : list or numpy.ndarray
        Receiver antenna orientation vector in the body frame of the satellite.
    rcver_max_offb_deg : float
        Maximum off-boresight angle for receiver's antenna [deg].
    sat_max_offb_deg : float
        Maximum Tx off-boresight angle for navigation satellites [deg].
    min_ctnr : float
        Minimum Carrier-to-Noise ratio required for signal acquisition [dB-Hz].
    """

    def __init__(self, data_path: Path, rcver_data_path: Path, export_base_path: Path,
                 rcver_antenna_dir_body: list, rcver_max_offb_deg: float, 
                 sat_max_offb_deg: float, min_ctnr: float):
        
        self.data_path = Path(data_path)
        self.rcver_data_path = Path(rcver_data_path)
        self.export_base_path = Path(export_base_path)
        self.rcver_antenna_dir_body = np.array(rcver_antenna_dir_body)
        self.rcver_max_offb_deg = rcver_max_offb_deg
        self.sat_max_offb_deg = sat_max_offb_deg
        self.min_ctnr = min_ctnr

        # Define internal path structures based on the project layout
        self.antennas_path = self.data_path / 'raw' / 'antennas'
        self.ephem_path = self.data_path / 'interim' / 'ephemeris'

        # Pre-load receiver trajectory and attitude data
        self._load_receiver_data()
        
        # Pre-load receiver gain pattern
        self.rcver_gain = np.loadtxt(self.antennas_path / 'receiver' / 'high_gain.txt', skiprows=1)

    def _load_receiver_data(self):
        """
        Loads the receiver's time, position, and calculates the vectorized 
        antenna direction in the GCRF frame based on the DCM.
        """
        # Load columns: JD, x, y, z, roll, pitch, yaw
        rcver_data = np.loadtxt(self.rcver_data_path, skiprows=1, usecols=(0, 13, 14, 15, 25, 26, 27)).T
        rcver_time_julian = rcver_data[0]
        
        rcver_time_dt = np.array([julian_to_datetime(jd) for jd in rcver_time_julian])
        self.rcver_time = np.array([(dt - rcver_time_dt[0]).total_seconds() for dt in rcver_time_dt])
        self.rcver_pos = 1e3 * np.column_stack((rcver_data[1], rcver_data[2], rcver_data[3]))

        # Vectorized Direction Cosine Matrix (DCM) calculation
        roll, pitch, yaw = rcver_data[4], rcver_data[5], rcver_data[6]
        N = len(roll)
        
        cx, sx = np.cos(roll), np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw), np.sin(yaw)

        Rx = np.zeros((N, 3, 3)); Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cx; Rx[:, 1, 2] = sx; Rx[:, 2, 1] = -sx; Rx[:, 2, 2] = cx
        Ry = np.zeros((N, 3, 3)); Ry[:, 0, 0] = cy; Ry[:, 0, 2] = -sy; Ry[:, 1, 1] = 1; Ry[:, 2, 0] = sy; Ry[:, 2, 2] = cy
        Rz = np.zeros((N, 3, 3)); Rz[:, 0, 0] = cz; Rz[:, 0, 1] = sz; Rz[:, 1, 0] = -sz; Rz[:, 1, 1] = cz; Rz[:, 2, 2] = 1

        # Matrix multiplication broadcasted over N steps: R = Rx @ Ry @ Rz
        rcver_dcm = Rx @ Ry @ Rz
        
        # Rotate the body vector into the GCRF frame
        body_vec = self.rcver_antenna_dir_body.reshape(3, 1)
        self.rcver_antenna_dir = np.squeeze(rcver_dcm @ body_vec)

    def _process_constellation(self, ephem_file: str, eirp_data: np.ndarray, freq: float) -> tuple:
        """
        Core logic to load ephemeris and compute visibility for a given set of satellites.
        """
        gnss_data = np.load(self.ephem_path / ephem_file)
        gnss_time_dt = [julian_to_datetime(jd + fr) for jd, fr in zip(gnss_data['jd'], gnss_data['fr'])]
        gnss_time_s = np.array([(dt - gnss_time_dt[0]).total_seconds() for dt in gnss_time_dt])
        
        gnss_pos = 1e3 * gnss_data['positions']
        gnss_vel = 1e3 * gnss_data['velocities']
        names = list(gnss_data['names'])

        access_list, ctnr_unf_list, ctnr_list, tx_list, rx_list = [], [], [], [], []

        for n, sat_pos in enumerate(gnss_pos):
            # Check if EIRP is a list of arrays (like GPS) or a single array
            current_eirp = eirp_data[n] if isinstance(eirp_data, list) else eirp_data

            access, ctnr_unf, ctnr, tx, rx = sat_visibility(
                self.rcver_time, self.rcver_pos, gnss_time_s, sat_pos, gnss_vel[n],
                self.rcver_antenna_dir, self.rcver_max_offb_deg, self.sat_max_offb_deg,
                self.min_ctnr, self.rcver_gain, current_eirp, freq
            )

            access_list.append(access)
            ctnr_unf_list.append(ctnr_unf)
            ctnr_list.extend(ctnr)
            tx_list.extend(tx)
            rx_list.extend(rx)

        return names, access_list, ctnr_unf_list, ctnr_list, tx_list, rx_list

    def run_gps(self):
        """Executes the visibility pipeline for the GPS constellation."""
        print("Calculating GPS visibility...")
        freq = 1575.42e6
        
        # Load EIRP blocks
        e_iir = np.loadtxt(self.antennas_path / "gnss" / "GPSIIR.txt", skiprows=1)
        e_iirm = np.loadtxt(self.antennas_path / "gnss" / "GPSIIRM.txt", skiprows=1)
        e_iif = np.loadtxt(self.antennas_path / "gnss" / "GPSIIF.txt", skiprows=1)
        e_iii = np.loadtxt(self.antennas_path / "gnss" / "GPSIII.txt", skiprows=1)
        
        # Note: This hardcoded block relies on the exact ordering of the Celestrak TLEs
        eirp_list = 6*[e_iir] + 7*[e_iirm] + 11*[e_iif] + 7*[e_iii]

        results = self._process_constellation('GPS.npz', eirp_list, freq)
        self._export_all('GPS', *results)

    def run_galileo(self):
        """Executes the visibility pipeline for the Galileo constellation."""
        print("Calculating Galileo visibility...")
        freq = 1575.42e6
        eirp = np.loadtxt(self.antennas_path / "gnss" / "GAL.txt", skiprows=1)
        results = self._process_constellation('Galileo.npz', eirp, freq)
        self._export_all('Galileo', *results)

    def run_glonass(self):
        """Executes the visibility pipeline for the GLONASS constellation."""
        print("Calculating GLONASS visibility...")
        freq = 1602e6
        eirp = np.loadtxt(self.antennas_path / "gnss" / "GPSIIRM.txt", skiprows=1) # Reusing GPSIIRM
        results = self._process_constellation('GLONASS.npz', eirp, freq)
        self._export_all('GLONASS', *results)

    def run_beidou(self):
        """Executes the visibility pipeline for the BeiDou constellation (MEO + GEO/IGSO)."""
        print("Calculating BeiDou visibility...")
        freq = 1575.42e6
        
        # Process MEO
        eirp_meo = np.loadtxt(self.antennas_path / "gnss" / "BDS_MEO.txt", skiprows=1)
        names_m, acc_m, c_unf_m, c_m, tx_m, rx_m = self._process_constellation('BDS_MEO.npz', eirp_meo, freq)

        # Process GEO/IGSO
        eirp_geo = np.loadtxt(self.antennas_path / "gnss" / "BDS_GEO.txt", skiprows=1)
        names_g, acc_g, c_unf_g, c_g, tx_g, rx_g = self._process_constellation('BDS_GEO_IGSO.npz', eirp_geo, freq)

        # Combine results
        self._export_all('BDS', names_m + names_g, acc_m + acc_g, c_unf_m + c_unf_g, 
                         c_m + c_g, tx_m + tx_g, rx_m + rx_g)

    def _export_all(self, const_name, names, access_data, ctnr_unfiltered, ctnr_vals, tx_angles, rx_angles):
        """Internal method to handle all file writing and directory creation."""
        out_dir = self.export_base_path / const_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Metadata / Config File
        with open(out_dir / 'access_data.out', 'w') as out:
            out.write(f"Receiver antenna max off-boresight angle [deg] = {self.rcver_max_offb_deg} \n")
            out.write(f"Receiver antenna orientation in body frame = {list(self.rcver_antenna_dir_body)} \n")
            out.write(f"Satellite antenna max off-boresight angle [deg] = {self.sat_max_offb_deg} \n")
            out.write(f"Minimum Carrier to Noise ratio = {self.min_ctnr} dB Hz\n")
            out.write(f"Receiver data path: {self.rcver_data_path.absolute()}\n")

        # 2. Raw Datasets
        np.savetxt(out_dir / 'ctnr.dat', ctnr_vals, fmt='%7.4f', header="C/N [dB Hz]", comments='')
        np.savetxt(out_dir / 'tx_angles.dat', tx_angles, fmt='%10.5e', header="Transmission off-boresight angle [deg]", comments='')
        np.savetxt(out_dir / 'rx_angles.dat', rx_angles, fmt='%10.5e', header="Reception off-boresight angles [deg]", comments='')

        # 3. Visible Satellites Count
        visible_sats = np.sum(access_data, axis=0)
        np.savetxt(out_dir / 'visible_sats.dat', np.column_stack((self.rcver_time, visible_sats)), 
                   fmt=['%22.16e', '%d'], delimiter='\t\t ', header="t[s] \t\t\t visible_sats", comments='')

        # 4. Access Matrix and Unfiltered C/N0
        header_str = "t[s]\t" + "\t\t ".join(names)
        
        fmt_access = ['%22.16e'] + ['%d'] * len(names)
        np.savetxt(out_dir / 'access_data.dat', np.column_stack((self.rcver_time, np.array(access_data).T)), 
                   fmt=fmt_access, delimiter='\t\t ', header=header_str, comments='')

        fmt_ctnr = ['%22.16e'] + ['%7.4f'] * len(names)
        np.savetxt(out_dir / 'ctnr_unfiltered.dat', np.column_stack((self.rcver_time, np.array(ctnr_unfiltered).T)), 
                   fmt=fmt_ctnr, delimiter='\t\t ', header=header_str, comments='')
        
        print(f" -> Export complete for {const_name}.")
