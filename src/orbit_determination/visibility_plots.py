"""
Module for visualizing GNSS visibility metrics, Link Budgets (C/N0), 
and outage distributions along a spacecraft's trajectory.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class VisibilityPlotter:
    """
    Ingests GNSS visibility data and generates analysis plots including C/N0 
    sensitivity, coverage statistics, and spatial access maps.

    Attributes
    ----------
    data_path : Path
        Path to the processed visibility data directory.
    constellations : list of str
        List of GNSS constellation names to process.
    rcver_pos : numpy.ndarray, optional
        Receiver position vectors [km], shape (N, 3).
    rcver_ta : numpy.ndarray, optional
        Receiver true anomaly over time [deg], shape (N,).
    plot_title : str
        Global title prefix for the generated plots.
    """

    def __init__(self, data_path: Path, constellations: list, 
                 rcver_pos: np.ndarray = None, rcver_ta: np.ndarray = None, 
                 plot_title: str = ""):
        self.data_path = Path(data_path)
        self.constellations = constellations
        self.rcver_pos = rcver_pos
        self.rcver_ta = rcver_ta
        self.plot_title = plot_title
        
        # Color mapping for constellations
        self.color_map = {
            "GPS": "tab:blue",
            "Galileo": "tab:green",
            "GLONASS": "tab:red",
            "BDS": "tab:orange"
        }

        # Data Caches
        self.time = None
        self.dt_sec = None
        self.access_data_all = []
        self.visible_sats_total = None
        self.ctnr_all = []
        self.tx_angles_all = []
        self.rx_angles_all = []
        
        # Dictionary to hold constellation-specific metrics
        self.const_data = {}

        self._load_data()

    def _load_data(self):
        """Loads and caches all visibility and link budget data from disk."""
        visible_sats_arrays = []

        for const in self.constellations:
            path = self.data_path / const
            print(f"Loading data for {const}...")
            
            # Load access data
            with open(path / 'access_data.dat', 'r') as f:
                line_names = f.readline().split()
                sat_names = line_names[1:]
                
                data = np.loadtxt(f)
                
            time = data[:, 0]
            access = data[:, 1:]
            vis_sats = np.sum(access, axis=1)
            
            if self.time is None:
                self.time = time
                self.dt_sec = np.diff(time, prepend=time[0])
                
            # Load angles and C/N0
            ctnr = np.loadtxt(path / 'ctnr.dat', skiprows=1)
            tx = np.loadtxt(path / 'tx_angles.dat', skiprows=1)
            rx = np.loadtxt(path / 'rx_angles.dat', skiprows=1)
            
            self.const_data[const] = {
                "names": sat_names,
                "access": access.T,
                "vis_sats": vis_sats,
                "ctnr": ctnr,
                "tx": tx,
                "rx": rx,
                "color": self.color_map.get(const, 'k')
            }
            
            self.access_data_all.extend(access.T)
            self.ctnr_all.extend(ctnr)
            self.tx_angles_all.extend(tx)
            self.rx_angles_all.extend(rx)
            visible_sats_arrays.append(vis_sats)

        self.visible_sats_total = np.sum(visible_sats_arrays, axis=0, dtype=np.int64)
        self.ctnr_all = np.array(self.ctnr_all)
        self.tx_angles_all = np.array(self.tx_angles_all)
        self.rx_angles_all = np.array(self.rx_angles_all)
        print(f"Data loaded successfully. Max total visible sats: {np.max(self.visible_sats_total)}")

    # --- Internal Math Utilities ---

    def _visibility_pot(self, visible_sats, num_sats):
        """Calculates percentage of time N satellites are visible."""
        dt_array = self.dt_sec
        sim_time = self.time[-1]
        pot = []
        for n in range(num_sats):
            mask = visible_sats >= n
            pot.append(100 * np.sum(dt_array[mask]) / sim_time)
        return pot

    def _find_perigee_apogee(self, ta):
        """Detects apogee and perigee indices from true anomaly."""
        perigee, apogee = [], []
        for i in range(1, len(ta)):
            if ta[i] < ta[i - 1]:
                perigee.append(i)
            if ta[i - 1] < 180 and ta[i] >= 180:
                apogee.append(i)
        return {'apogee': apogee, 'perigee': perigee}

    def _coverage_outage_windows(self, visible_sats, dt, min_sats):
        """Calculates durations of consecutive coverage and outages."""
        coverage = (visible_sats >= min_sats).astype(int)
        outage_wins, coverage_wins = [], []
        c_dur, o_dur = 0, 0
        
        for i, has_cov in enumerate(coverage):
            if not has_cov:
                o_dur += dt[i]
                if c_dur > 0:
                    coverage_wins.append(c_dur)
                    c_dur = 0
            else:
                c_dur += dt[i]
                if o_dur > 0:
                    outage_wins.append(o_dur)
                    o_dur = 0
                    
        if c_dur > 0: coverage_wins.append(c_dur)
        if o_dur > 0: outage_wins.append(o_dur)
        
        return coverage_wins or [0], outage_wins or [0]

    def _calculate_cdf(self, data):
        """Calculates the Empirical CDF of a dataset, ignoring 0-duration placeholders."""
        # Convert to numpy array and filter out 0-duration (no outage) placeholders
        data = np.array(data)
        data = data[data > 0] 
        
        # Safety guard: if there are no real outages, return empty arrays safely
        if len(data) == 0:
            return np.array([]), np.array([])
            
        x, counts = np.unique(data, return_counts=True)
        cusum = np.cumsum(counts)
        return x, cusum / cusum[-1]

    # --- Plotting Methods ---

    def plot_ctnr_distribution(self, min_ctnr=20, save_path=None):
        """Plots the histogram of C/N0 values and highlights discarded signals."""
        fig, ax = plt.subplots(figsize=(8, 5))
        n, bins, patches = ax.hist(self.ctnr_all, bins=100, density=True, 
                                   edgecolor='tab:blue', linewidth=1.2)
        
        for i, p in enumerate(patches):
            if bins[i] < min_ctnr:
                p.set_facecolor('none')
            else:
                p.set_facecolor('tab:blue')

        area_discarded = np.mean(self.ctnr_all < min_ctnr) * 100
        patch_used = mpatches.Patch(facecolor='tab:blue', edgecolor='tab:blue', 
                                    label=rf"$C/N_{{0}} \geq {min_ctnr}$ dB-Hz")
        patch_discard = mpatches.Patch(facecolor='none', edgecolor='tab:blue', 
                                       label=f"Discarded signals ({area_discarded:.1f}%)")

        ax.set_xlabel(r"$C/N_{0}$ [dB Hz]")
        ax.set_ylabel("Probability density")
        ax.legend(handles=[patch_used, patch_discard])
        ax.set_title(self.plot_title + " - C/N0 distribution")
        
        if save_path:
            plt.savefig(Path(save_path) / 'ctnr_dist.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_ctnr_sensitivity(self, ref_ctnr=20, save_path=None):
        """Plots the percentage of discarded measurements vs C/N0 threshold."""
        thresholds = np.linspace(np.min(self.ctnr_all), np.max(self.ctnr_all), 200)
        discarded = [np.mean(self.ctnr_all < u) * 100 for u in thresholds]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, discarded, color='tab:red', linewidth=2)

        # Reference Points
        points = [ref_ctnr, 25, 30, 35]
        for p in points:
            area = np.mean(self.ctnr_all < p) * 100
            ax.plot(p, area, marker='o', color='tab:blue', markersize=8, zorder=5)
            ax.annotate(f"{p} dB Hz $\\rightarrow$ {area:.1f}%", 
                        xy=(p, area), xytext=(15, -25), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:blue", alpha=0.9),
                        fontsize=10, color='tab:blue')

        ax.set_xlabel(r"Minimum threshold of $C/N_{0}$ [dB Hz]")
        ax.set_ylabel("Discarded signals [%]")
        ax.set_ylim(0, 105)
        ax.set_xlim(np.min(thresholds), np.max(thresholds))
        
        if save_path:
            plt.savefig(Path(save_path) / 'ctnr_sensitivity.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_cumulative_visibility(self, save_path=None):
        """Plots the percentage of time N satellites are visible."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        total_sats = len(self.access_data_all)
        pot_total = self._visibility_pot(self.visible_sats_total, total_sats)
        
        if len(self.constellations) > 1:
            ax.plot(range(total_sats), pot_total, marker='o', color='k',
                    label=f"Multi-GNSS ({pot_total[1]:.0f}%)" + r"$_{\geq 1}$" + f" ({pot_total[4]:.0f}%)" + r"$_{\geq 4}$")
            
        for const in self.constellations:
            c_data = self.const_data[const]
            num_c_sats = len(c_data["access"])
            pot = self._visibility_pot(c_data["vis_sats"], num_c_sats)
            ax.plot(range(num_c_sats), pot, marker='o', color=c_data["color"],
                    label=f"{const} ({pot[1]:.0f}%)" + r"$_{\geq 1}$" + f" ({pot[4]:.0f}%)" + r"$_{\geq 4}$")

        ax.set_xlabel(r"$N_{sat}$")
        ax.set_ylabel(r"Time [%] with $\geq N_{sat}$ visible satellites")
        ax.set_xlim(0, 20)
        ax.legend()
        ax.set_title(self.plot_title)
        
        if save_path:
            plt.savefig(Path(save_path) / 'cumulative_visibility.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_outage_cdf(self, save_path=None):
        """Plots the Empirical CDF of outage durations per revolution."""
        if self.rcver_ta is None:
            print("Error: rcver_ta is required to plot outage CDF over revolutions.")
            return

        crossings = self._find_perigee_apogee(self.rcver_ta)
        apogees = crossings['apogee']
        
        outages_1sat, outages_4sat = [], []
        
        for n in range(len(apogees) - 1):
            start, end = apogees[n], apogees[n+1]
            vis_rev = self.visible_sats_total[start:end]
            dt_min_rev = self.dt_sec[start:end] / 60.0
            
            _, o_1 = self._coverage_outage_windows(vis_rev, dt_min_rev, 1)
            _, o_4 = self._coverage_outage_windows(vis_rev, dt_min_rev, 4)
            outages_1sat.extend(o_1)
            outages_4sat.extend(o_4)

        val_1, cdf_1 = self._calculate_cdf(outages_1sat)
        val_4, cdf_4 = self._calculate_cdf(outages_4sat)

        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot 0-satellite outages (Complete Signal Loss)
        if len(val_1) > 0:
            ax.plot(val_1, cdf_1, drawstyle='steps-post', color='tab:blue', label=r'$N_{sats} = 0$')
        else:
            # Add a text annotation if coverage never dropped to 0
            ax.plot([], [], color='tab:blue', label=r'$N_{sats} = 0$ (100% Coverage!)')

        # Plot <4-satellite outages (Navigational Loss)
        if len(val_4) > 0:
            ax.plot(val_4, cdf_4, drawstyle='steps-post', color='tab:red', label=r'$N_{sats} < 4$')
        else:
            # Add a text annotation if coverage never dropped below 4
            ax.plot([], [], color='tab:red', label=r'$N_{sats} < 4$ (100% Coverage!)')

        # If there were absolutely no outages at all, put a large success text in the middle
        if len(val_1) == 0 and len(val_4) == 0:
            ax.text(0.5, 0.5, '100% GNSS Coverage\n(No outages detected)', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=14, color='tab:green', 
                    bbox=dict(facecolor='white', edgecolor='tab:green', pad=10.0))
            ax.set_xlim(0, 10) # Set a dummy x-axis scale so the plot renders nicely

        ax.set_xlabel("Outage duration [min]")
        ax.set_ylabel("Cumulative distribution")
        ax.legend()
        ax.set_title(self.plot_title + " - Signal outage duration")
        
        if save_path:
            plt.savefig(Path(save_path) / 'outage_cdf.jpg', dpi=300, bbox_inches='tight')
        plt.show()
