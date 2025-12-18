import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
import treecorr
import glass
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors

class DefectFinder:
    """Class to find topological defects in shear orientation field."""

    def __init__(self, data_path):
        """Initialize the DefectFinder with shear orientation data."""
        data = np.load(data_path)
        self.angle_deg = data['angle_deg']
        self.kappa = data['kappa']
        self.gamm1 = data['gamm1']
        self.gamm2 = data['gamm2']
        self.phi = data['shear_angles']
        self.shear_amp = np.sqrt(self.gamm1**2 + self.gamm2**2)
        
        # Determine if data is flat sky (2D) or Healpix (1D)
        if self.phi.ndim == 2:
            self.is_flat = True
            self.shape = self.phi.shape
            self.nside = None
        else:
            self.is_flat = False
            self.nside = hp.npix2nside(len(self.phi))
            
        # Alias for compatibility
        self.phi_bar = self.phi
        self.kappa_bar = self.kappa

    def load_defects(self, defect_datapath):
        """Load defects from a saved npz file."""
        data = np.load(defect_datapath, allow_pickle=True)
        self.defects = data['defects'].item()
        #print(f"Loaded {len(self.defects)} defects from {defect_datapath}")

    def delta_phi(self, phi1s, phi2s):
        """Compute the difference between two angles, handling periodicity."""
        diffs = phi2s - phi1s
        betas = np.zeros_like(diffs)
        betas[diffs > np.pi/2] = -np.pi
        betas[diffs < -np.pi/2] = np.pi
        dphis = diffs + betas
        return dphis

    def winding_number(self, pixs=None):
        """Compute the total delta phi around the 8 neighbors of a pixel, add up counter-clockwise."""
        if self.is_flat:
            return self._winding_number_flat()
        else:
            return self._winding_number_healpix(pixs)

    def _winding_number_healpix(self, pixs):
        if pixs is None:
            pixs = np.arange(len(self.phi))
            
        nside = self.nside
        neighbors = hp.get_all_neighbours(nside, pixs) # 8xn_pixs array
        # remove invalid neighbors (at the boundary)
        valid_cols = np.all(neighbors >= 0, axis=0)
        valid_pixs = pixs[valid_cols]
        valid_neighbors = neighbors[:, valid_cols]
        dphi_totals = np.zeros(len(pixs))
        for i in range(8):
            pix1 = valid_neighbors[i,:]
            pix2 = valid_neighbors[(i+1) % 8,:]
            dphis = self.delta_phi(self.phi[pix1], self.phi[pix2])
            dphi_totals[valid_cols] += dphis
        wns = dphi_totals / (2 * np.pi)  # in units of 2 pi
        # avoid numerical fluctuations around half-integers
        wns = np.round(wns * 2) / 2
        return wns

    def _winding_number_flat(self):
        """Compute winding number for 2D flat sky map."""
        rows, cols = self.shape
        
        # Define 8 neighbors in counter-clockwise order
        # (row_offset, col_offset)
        # Right: (0, 1), Top-Right: (-1, 1), Top: (-1, 0), Top-Left: (-1, -1)
        # Left: (0, -1), Bottom-Left: (1, -1), Bottom: (1, 0), Bottom-Right: (1, 1)
        neighbor_offsets = [
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
            (0, -1), (1, -1), (1, 0), (1, 1)
        ]
        
        # Pad with NaNs to handle boundaries
        padded_phi = np.pad(self.phi, 1, mode='constant', constant_values=np.nan)
        
        neighbors = []
        for dr, dc in neighbor_offsets:
            # Slice the padded array to get the neighbor view
            # Center is at [1:rows+1, 1:cols+1]
            # Neighbor is at [1+dr:rows+1+dr, 1+dc:cols+1+dc]
            n_view = padded_phi[1+dr : rows+1+dr, 1+dc : cols+1+dc]
            neighbors.append(n_view)
            
        dphi_total = np.zeros(self.shape)
        
        for i in range(8):
            n1 = neighbors[i]
            n2 = neighbors[(i+1)%8]
            dphis = self.delta_phi(n1, n2)
            dphi_total += dphis
            
        wns = dphi_total / (2 * np.pi)
        wns = np.round(wns * 2) / 2
        
        # Set boundary/invalid pixels to 0
        wns[np.isnan(wns)] = 0
        
        return wns

    def find_defects(self, savepath=None, cluster=True):
        """Find defects in the shear orientation field."""
        self.defects = {}
        
        if self.is_flat:
            wnums = self.winding_number()
            rows, cols = np.where(np.abs(wnums) >= 0.5)
            for r, c in zip(rows, cols):
                self.defects[(r, c)] = wnums[r, c]
                
            if cluster:
                self._cluster_defects_flat()
        else:
            npix = len(self.phi)
            wnums = self.winding_number(np.arange(npix))
            defect_indices = np.where(np.abs(wnums) >= 0.5)[0]
            self.defects = dict(zip(defect_indices, wnums[defect_indices]))
            if cluster:
                self._cluster_defects_healpix()
                
        if savepath:
            np.savez(savepath, defects=self.defects)
        #print(f"Found {len(self.defects)} defects, saving to {savepath}")
        return self.defects
    
    def _cluster_defects_flat(self):
        defect_pixels = list(self.defects.keys())
        # Sort defects by shear amplitude (ascending)
        defect_pixels.sort(key=lambda p: self.shear_amp[p])

        clustered_defects = {}
        visited = set()
        
        for pix in defect_pixels:
            if pix in visited:
                continue
            wnum = self.defects[pix]
            clustered_defects[pix] = wnum
            
            r, c = pix
            # Mark 3x3 neighbors as visited
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    visited.add((r+dr, c+dc))
        self.defects = clustered_defects

    def _cluster_defects_healpix(self):
        defect_pixels = list(self.defects.keys())
        # Sort defects by shear amplitude (ascending)
        defect_pixels.sort(key=lambda p: self.shear_amp[p])

        clustered_defects = {}
        visited = set()
        for pix in defect_pixels:
            if pix in visited:
                continue
            wnum = self.defects[pix]
            clustered_defects[pix] = wnum
            neighbors = hp.get_all_neighbours(self.nside, pix)
            for neighbor in neighbors:
                visited.add(neighbor)
        self.defects = clustered_defects

    def get_coords(self, pix):
        if self.is_flat:
            r, c = pix
            if self.angle_deg.ndim == 3:
                ra = self.angle_deg[0, r, c]
                dec = self.angle_deg[1, r, c]
            else:
                # Fallback if angle_deg is not 3D meshgrid
                ra = c 
                dec = r
            return ra, dec
        else:
            theta, phi = hp.pix2ang(self.nside, pix)
            ra = phi * 180/np.pi
            dec = (np.pi/2 - theta) * 180/np.pi
            return ra, dec

    def nearest_distance(self, charge, k=1):
        """Compute the nearest neighbor distance distribution for defects with given charge. Unit: arcmin."""
        ra_list = []
        dec_list = []
        for pix in self.defects.keys():
            wnum = self.defects[pix]
            if wnum == charge:
                ra, dec = self.get_coords(pix)
                ra_list.append(ra)
                dec_list.append(dec)
        
        if not ra_list:
            return np.array([])
            
        ra_arr = np.array(ra_list)
        dec_arr = np.array(dec_list)
        
        coords = np.vstack((ra_arr, dec_arr)).T
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        distances = distances * 60  # convert to arcmin
        
        if distances.shape[1] > k:
            nearest_distances = distances[:, k]
            return nearest_distances
        else:
            return np.array([])

    def nearest_opposite_distance(self, charge, k=1):
        """Compute the nearest neighbor distance distribution between +1/2 and -1/2 defects."""
        ra_pos = []
        dec_pos = []
        ra_neg = []
        dec_neg = []
        for pix in self.defects.keys():
            wnum = self.defects[pix]
            ra, dec = self.get_coords(pix)
            if wnum == 0.5:
                ra_pos.append(ra)
                dec_pos.append(dec)
            elif wnum == -0.5:
                ra_neg.append(ra)
                dec_neg.append(dec)
                
        ra_pos = np.array(ra_pos)
        dec_pos = np.array(dec_pos)
        ra_neg = np.array(ra_neg)
        dec_neg = np.array(dec_neg)
        
        if len(ra_pos) == 0 or len(ra_neg) == 0:
            return np.array([])
        
        coords_pos = np.vstack((ra_pos, dec_pos)).T
        coords_neg = np.vstack((ra_neg, dec_neg)).T
        
        if charge == 0.5:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords_neg)
            distances, indices = nbrs.kneighbors(coords_pos)
        else:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords_pos)
            distances, indices = nbrs.kneighbors(coords_neg)
            
        distances = distances * 60  # convert to arcmin
        if distances.shape[1] >= k:
            nearest_distances = distances[:, k-1]
            return nearest_distances
        else:
            return np.array([])

    def convergence_dist(self, charge):
        """Compute the convergence value distribution at defect locations with given charge."""
        kappa_values = []
        for pix in self.defects.keys():
            wnum = self.defects[pix]        
            if wnum == charge:
                kappa = self.kappa_bar[pix]
                kappa_values.append(kappa)
        kappa_values = np.array(kappa_values)
        return kappa_values

    def convergence_correlation(self, min_sep=1., max_sep=100., n_bins=10, mask=None):
        """Compute the convergence auto-correlation function.
        
        Args:
            min_sep: Minimum separation in arcmin.
            max_sep: Maximum separation in arcmin.
            n_bins: Number of bins.
            mask: Optional boolean mask (same shape as kappa) to restrict calculation to a specific region.
        """
        kappa_all = self.kappa
        
        if self.is_flat:
            # Flat sky case
            if mask is None:
                mask = np.ones(self.shape, dtype=bool)
            
            # Filter NaNs and apply mask
            valid_mask = mask & (~np.isnan(kappa_all))
            rows, cols = np.where(valid_mask)
            kappa_valid = kappa_all[rows, cols]
            
            if self.angle_deg.ndim == 3:
                # Use provided RA/Dec grids
                ra_valid = self.angle_deg[0, rows, cols]
                dec_valid = self.angle_deg[1, rows, cols]
                cat_kappa = treecorr.Catalog(ra=ra_valid, dec=dec_valid, k=kappa_valid, ra_units='degrees', dec_units='degrees')
            else:
                # Fallback to pixel coordinates (assuming pixels correspond to arcmin or similar units)
                # Note: This fallback might need adjustment depending on your specific flat sky pixel scale
                cat_kappa = treecorr.Catalog(x=cols, y=rows, k=kappa_valid, x_units='arcmin', y_units='arcmin')
        else:
            # Healpix case
            # Filter out masked values (Healpix UNSEEN) and NaNs for partial sky support
            valid_mask = (kappa_all != hp.UNSEEN) & (~np.isnan(kappa_all))
            
            if mask is not None:
                valid_mask = valid_mask & mask
                
            valid_indices = np.where(valid_mask)[0]
            kappa_valid = kappa_all[valid_indices]
            
            theta_valid, phi_valid = hp.pix2ang(self.nside, valid_indices)
            ra_valid = phi_valid * 180/np.pi
            dec_valid = (np.pi/2 - theta_valid) * 180/np.pi
            
            cat_kappa = treecorr.Catalog(ra=ra_valid, dec=dec_valid, k=kappa_valid, ra_units='degrees', dec_units='degrees')
            
        kk = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
        kk.process(cat_kappa)
        thetas = kk.meanr
        xi = kk.xi
        return thetas, xi
