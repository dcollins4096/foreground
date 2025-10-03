import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

# --------------------------
# 1. Download Planck maps (example: 2018 PR3 Commander maps)
# --------------------------
# You can replace with other channels (HFI: 100, 143, 217, 353 GHz, LFI: 30, 44, 70 GHz)
if 0:
    planck_urls = {
        "30GHz": "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "44GHz": "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=LFI_SkyMap_044-BPassCorrected_1024_R2.01_full.fits",
        "70GHz": "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=LFI_SkyMap_070-BPassCorrected_1024_R2.01_full.fits",
        "100GHz": "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=HFI_SkyMap_100_2048_R3.01_full.fits",
        "143GHz": "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=HFI_SkyMap_143_2048_R3.01_full.fits",
        "217GHz": "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=HFI_SkyMap_217_2048_R3.01_full.fits",
        "353GHz": "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=HFI_SkyMap_353_2048_R3.01_full.fits",
    }

    data_dir = "planck_maps"
    os.makedirs(data_dir, exist_ok=True)

    def download_maps():
        local_files = {}
        for name, url in planck_urls.items():
            outpath = os.path.join(data_dir, f"{name}.fits")
            if not os.path.exists(outpath):
                print(f"Downloading {name}...")
                urlretrieve(url, outpath)
            local_files[name] = outpath
        return local_files

    maps = download_maps()
maps = {
30:"data/LFI_SkyMap_030_1024_R2.01_full.fits",
#44:"data/LFI_SkyMap_044_1024_R2.01_full.fits",
#70:"data/LFI_SkyMap_070_1024_R2.01_full.fits",
#70:"data/LFI_SkyMap_070_2048_R2.01_full.fits",
100:"data/HFI_SkyMap_100_2048_R2.02_full.fits"}
#143:"data/HFI_SkyMap_143_2048_R2.02_full.fits",
#217:"data/HFI_SkyMap_217_2048_R2.02_full.fits",
#353:"data/HFI_SkyMap_353_2048_R2.02_full.fits",
#545:"data/HFI_SkyMap_545_2048_R2.02_full.fits",
#857:"data/HFI_SkyMap_857_2048_R2.02_full.fits"}


# --------------------------
# 2. Load maps into healpy
# --------------------------
sky_maps = {}
for name, path in maps.items():
    try:
        print('load map',path)
        m = hp.read_map(path, field=0)
        sky_maps[name] = m
        print(f"Loaded {name} with Nside={hp.get_nside(m)}")
    except Exception as e:
        print(f"Failed to load {name}: {e}")

# --------------------------
# 3. Quick visualization
# --------------------------
print('plot')
plt.figure(figsize=(10, 5))
hp.mollview(sky_maps[100], title="Planck 143 GHz", norm="hist", sub=(1,1,1))
print('save')
plt.savefig('test2.png')
#plt.show()

if 0:
# --------------------------
# 4. Simple Foreground Removal Tools
# --------------------------

    def simple_ilc(maps_dict):
        """
        Internal Linear Combination (ILC): 
        minimize variance subject to unit response to CMB.
        """
        freqs = list(maps_dict.keys())
        maps_arr = np.array([maps_dict[f] for f in freqs])
        npix = maps_arr.shape[1]
        
        # covariance
        cov = np.cov(maps_arr)
        invcov = np.linalg.pinv(cov)
        
        # weights (unit response vector = 1)
        ones = np.ones(len(freqs))
        w = invcov @ ones / (ones @ invcov @ ones)
        
        cmb_map = np.tensordot(w, maps_arr, axes=(0,0))
        return cmb_map, w

    cmb_est, weights = simple_ilc(sky_maps)

    print("ILC Weights:", dict(zip(sky_maps.keys(), weights)))

    hp.mollview(cmb_est, title="CMB Estimate (Simple ILC)", norm="hist")
    plt.show()

# --------------------------
# 5. Template subtraction example (e.g. remove dust with 353 GHz map)
# --------------------------
    dust_template = sky_maps["353GHz"]
    cmb_143 = sky_maps["143GHz"] - 0.02 * dust_template  # scaling is arbitrary
    hp.mollview(cmb_143, title="143 GHz with dust template subtraction", norm="hist")
    plt.show()
