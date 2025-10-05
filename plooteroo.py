import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm, Normalize,SymLogNorm

mapfile = "data/HFI_SkyMap_353_2048_R2.02_full.fits"
if 'm' not in dir():
    m = hp.read_map(mapfile, field=0, verbose=False)
    Q = hp.read_map(mapfile, field=1, verbose=False)
    U = hp.read_map(mapfile, field=2, verbose=False)
    mE = hp.read_map("E_map.fits", field=0, verbose=False)
    mB = hp.read_map("B_map.fits", field=0, verbose=False)

# Taurus Galactic center
l_taurus, b_taurus = 170.0, -15.0

# Extract cutout as an array (not just a plot)
cutout = hp.gnomview(m, rot=(l_taurus, b_taurus),
                     xsize=800, reso=5.0,
                     return_projected_map=True,
                     no_plot=True)   # suppress auto-plot
cutoutE = hp.gnomview(mE, rot=(l_taurus, b_taurus),
                     xsize=800, reso=5.0,
                     return_projected_map=True,
                     no_plot=True)   # suppress auto-plot
cutoutB = hp.gnomview(mB, rot=(l_taurus, b_taurus),
                     xsize=800, reso=5.0,
                     return_projected_map=True,
                     no_plot=True)   # suppress auto-plot

# Mask negative values (log scale needs >0)
cutout = np.ma.masked_less_equal(cutout, 0)

# Plot manually with log scaling
fig,ax=plt.subplots(1,3,figsize=(12,4))
im = ax[0].imshow(cutout, origin="lower",
                norm=LogNorm(vmin=cutout.min(), vmax=cutout.max()),
                cmap="inferno")  # pick your favorite colormap
im = ax[1].imshow(cutoutE, origin="lower",
                norm=Normalize(vmin=cutoutE.min(), vmax=cutoutE.max()),
                cmap="inferno")  # pick your favorite colormap
im = ax[2].imshow(cutoutB, origin="lower",
                norm=SymLogNorm(1e-4,vmin=cutoutB.min(), vmax=cutoutB.max()),
                cmap="inferno")  # pick your favorite colormap
fig.savefig('%s/plots/test1'%os.environ['HOME'])

