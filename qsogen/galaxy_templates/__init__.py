import numpy as np
import os
path = os.path.dirname(__file__)+'/'

elliptical_13gyr = np.genfromtxt(path+"elliptical_swire_13gyr.sed",unpack=True)
elliptical_5gyr = np.genfromtxt(path+"elliptical_swire_5gyr.sed",unpack=True)
elliptical_2gyr = np.genfromtxt(path+"elliptical_swire_2gyr.sed",unpack=True)
s0 = np.genfromtxt(path+"spiral_swire_s0.sed",unpack=True)
spiral_sa = np.genfromtxt(path+"spiral_swire_sa.sed",unpack=True)
spiral_sb = np.genfromtxt(path+"spiral_swire_sb.sed",unpack=True)
spiral_sc = np.genfromtxt(path+"spiral_swire_sc.sed",unpack=True)
spiral_sd = np.genfromtxt(path+"spiral_swire_sd.sed",unpack=True)
spiral_sdm = np.genfromtxt(path+"spiral_swire_sdm.sed",unpack=True)