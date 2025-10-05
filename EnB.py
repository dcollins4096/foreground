# Requires: healpy, astropy, numpy
# pip install healpy astropy numpy

import healpy as hp
import numpy as np
from astropy.io import fits

def read_planck_qu_maps(fname, q_field=1, u_field=2, nest=False):
    """
    Read Q and U from a Planck-like HEALPix FITS file.
    - fname: path to fits
    - q_field, u_field: indices (0-based) of the Q and U columns/fields in the FITS map
      (adjust these to match the file you have; some Planck maps have multiple columns)
    - nest: whether to reorder to NESTED (False means keep RING ordering)
    Returns (Q, U, header) arrays in RING ordering unless nest=True requested.
    """
    # healpy.read_map is flexible and accepts a list of fields; it also auto-handles UNSEEN
    # but some Planck maps use columns; you may need to inspect the fits HDU.
    maps = hp.read_map(fname, field=[q_field, u_field], verbose=False)
    Q, U = maps[0], maps[1]
    if nest:
        # convert to RING (most healpy functions expect RING by default)
        Q = hp.pixelfunc.reorder(Q, n2r=True)
        U = hp.pixelfunc.reorder(U, n2r=True)
    return Q, U

def qu_to_eb_maps(Q, U, lmax=None, iter=3):
    """
    Convert Q,U maps to E and B maps (returning maps and alms).
    - Q,U: healpix arrays (RING ordering), can contain hp.UNSEEN or np.nan
    - lmax: maximum multipole; if None uses 3*nside-1
    Returns: E_map, B_map, almE, almB
    """
    # Replace NaNs with hp.UNSEEN (or mask) to avoid transform errors:
    Q = np.where(np.isfinite(Q), Q, hp.UNSEEN)
    U = np.where(np.isfinite(U), U, hp.UNSEEN)

    nside = hp.get_nside(Q)
    if lmax is None:
        lmax = 3 * nside - 1

    # map2alm_spin returns two alm arrays for spin s (see healpy docs)
    # For spin=2 and input [Q, U] you get alms that correspond to E and B.
    print('alm')
    almE, almB = hp.sphtfunc.map2alm_spin([Q, U], spin=2, lmax=lmax, mmax=None)

    # Convert back to maps (option A). Using alm2map will produce scalar maps from each alm.
    print('map')
    E_map = hp.alm2map(almE, nside, lmax=lmax)
    B_map = hp.alm2map(almB, nside, lmax=lmax)

    # Alternative: if you want spin maps from E/B alms (to recreate Q/U) use:
    # q_recon, u_recon = hp.sphtfunc.alm2map_spin([almE, almB], nside, spin=2, lmax=lmax)

    return E_map, B_map, almE, almB

# Example usage:
if __name__ == "__main__":
    planck_fits = "data/HFI_SkyMap_353_2048_R2.02_full.fits"   # replace with your file
    print('read')
    Q, U = read_planck_qu_maps(planck_fits, q_field=1, u_field=2, nest=False)
    print('done')

    # optionally subtract monopole/mean / demean if needed:
    # Q = hp.remove_monopole(Q)  # or just np.nanmean over valid pixels

    E_map, B_map, almE, almB = qu_to_eb_maps(Q, U)

    # Save results
    hp.write_map("E_map.fits", E_map, overwrite=True)
    hp.write_map("B_map.fits", B_map, overwrite=True)
    # Save alms (healpy.almxfl / hp.write_alm could be used; healpy has write_alm)
    hp.write_alm( "almE.fits", almE,overwrite=True)
    hp.write_alm( "almB.fits", almB,overwrite=True)

    print("Done: wrote E_map.fits, B_map.fits, almE.fits, almB.fits")

