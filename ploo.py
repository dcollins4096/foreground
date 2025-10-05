import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# --- Parameters ---
mapfile = "data/HFI_SkyMap_353_2048_R2.02_full.fits"  # polarization map (353 GHz)
l_taurus, b_taurus = 170.0, -15.0
reso = 5.0     # arcmin per pixel
xsize = 400    # pixels in cutout

# --- Load I, Q, U maps ---
if 'I_map' not in dir():
    I_map, Q_map, U_map = hp.read_map(mapfile, field=(0,1,2), verbose=False)

# --- Extract gnomonic projections (Taurus patch) ---
I_cut = hp.gnomview(I_map, rot=(l_taurus, b_taurus),
                    xsize=xsize, reso=reso, return_projected_map=True, no_plot=True)
Q_cut = hp.gnomview(Q_map, rot=(l_taurus, b_taurus),
                    xsize=xsize, reso=reso, return_projected_map=True, no_plot=True)
U_cut = hp.gnomview(U_map, rot=(l_taurus, b_taurus),
                    xsize=xsize, reso=reso, return_projected_map=True, no_plot=True)

# --- Polarization quantities ---
P = np.sqrt(Q_cut**2 + U_cut**2)
psi = 0.5 * np.arctan2(U_cut, Q_cut)       # polarization angle
psi_B = psi + np.pi/2                      # rotate 90° for B-field

# --- Downsample for clarity (optional) ---
step = 10
Y, X = np.mgrid[0:I_cut.shape[0], 0:I_cut.shape[1]]
Xq, Yq = X[::step, ::step], Y[::step, ::step]
Uq = np.cos(psi_B)[::step, ::step]
Vq = np.sin(psi_B)[::step, ::step]

# --- Plot intensity + B-field lines ---
plt.figure(figsize=(7,7))
plt.imshow(I_cut, origin="lower", cmap="inferno", norm=LogNorm(vmin=max(I_cut.min(),1e-6), vmax=I_cut.max()))
plt.colorbar(label="Intensity (K_CMB)")
#plt.quiver(Xq, Yq, Uq, Vq, color="cyan", scale=30, headwidth=0, headlength=0, headaxislength=0, minlength=0)
plt.streamplot(X, Y, np.cos(psi_B), -np.sin(psi_B),
                              color="cyan", density=1.2, linewidth=0.6, arrowsize=0)
plt.title("Taurus Molecular Cloud – Magnetic Field (353 GHz)")
plt.xlabel("X [pixels]")
plt.ylabel("Y [pixels]")
import os
plt.savefig('%s/plots/Fields'%os.environ['HOME'])

