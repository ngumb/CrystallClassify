
import drprobe as drp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpli

import cv2
import os
from copy import deepcopy
import pandas as pd
import emilys.optics.econst as ec
from emilys.structure.supercell import supercell
import emilys.structure.celio as fcel
from scipy.spatial.transform import Rotation as R

"""

make a supercell with a cluster of randomly placed atoms

"""


ImgNum =  10000 # Number of images to simulate
Maindir = "C:/Au-NPs" # Location to store the simulation files and subfolders

ht = 300
nx = 512
ny = 512
nz = 64
a = 6.06208
b = a

NPStructures = ["amorph"]

for inp in range(0,len(NPStructures)):
    for j in range(6192,ImgNum):
        NPStructure = NPStructures[inp]
        
        NPFile = Maindir + "/Raw_pbd/" + NPStructure + ".txt"
        
        NPsize =  1+np.random.uniform()*3
        focus = 1+(2*np.random.uniform()-1)*5
        astx = (2*np.random.uniform()-1)*3
        asty = (2*np.random.uniform()-1)*3
        focus_spread = 5+(2*np.random.uniform()-1)*0.5
        
   
        FileName = NPStructure + '_size{:.0f}'.format(NPsize*10) + '_j{:.0f}'.format(j)
        cel_file = Maindir + "/cel/" + FileName + ".cel"
        slice_name = Maindir + "/slc/Au" #+ FileName
        
        BoxD = 10 #Distance of the particle to the box edges in a
        
        # Box size for the particle to be placed into
        box = np.array([a, a, a])
        # make a supercell for a gold particle
        sc1 = supercell()
        sc1.a0 = box.copy() * 10
        Zp = 79 # atomic number (Au)
        uiso = 0.007497768 # for Gold
       
        # make a supercell with a cluster of randomly placed atoms (Au)
        
        dnp = np.array([NPsize, NPsize, NPsize]) # particle rms diameters [nm]
        
        pnp = (2+dnp/2)/a # particle center position (fract)
       
        Mp = 196.96657 * ec.PHYS_MASSU # atomic mass [kg] (Au)
        Rp = 0.248 # bond length [nm]
        rhop = 19.3 # material density [g cm^-3] [10^-3 * 10^-21 kg * nm^-3]
        Vp0 = np.product(dnp) # approx. particle volume [nm^3]
        Np0 = int(1.E-24 * rhop / Mp * Vp0) # approx. number of atoms
        print('adding initial', Np0, 'atoms')
        ip = 0
        while ip < Np0:
            pf = 2.*(np.random.uniform(size=3) - 0.5)
            while (np.dot(pf,pf) > 1.):
                pf = 2.*(np.random.uniform(size=3) - 0.5)
            f = pnp + 0.5 * pf * dnp / box
            sc1.add_atom(Zp, uiso, f)
            ip += 1
        l_at_all = list(np.arange(0, Np0, dtype=int))
        l_at_rem = sc1.remove_close_atoms(l_at_all, Rp*10.)
        print('removing', len(l_at_rem), 'too close atoms')
        sc1.delete_atoms(l_at_rem)
        
        
        
        # make a supercell with a foil of randomly placed atoms (C)
        sc2 = supercell()
        sc2.a0 = box.copy() * 10.
        # Foil position parameters
        dfoil = 1+np.random.uniform()*2 # foil thickness [nm]
        pfoil = (2+dnp[2]+dfoil/2)/a # foil center position (fract)
        
        if j<=ImgNum/2:
            Zf = 14 # atomic number (Si)
            Mf = 28.0855 * ec.PHYS_MASSU # atomic mass [kg] (Si) #12.011 * ec.PHYS_MASSU # atomic mass [kg] (C)# 28.0855 * ec.PHYS_MASSU # atomic mass [kg] (Si) #
            Rf = 0.160 # bond length [nm]
            rhof = 2.33 # Si 1.8 # C material density [g cm^-3] [10^-3 * 10^-21 kg * nm^-3] #2.33 # Si
            Vf0 = dfoil * box[0] * box[1] # approx. foil volume [nm^3]
            Nf0 = int(1.E-24 * rhof / Mf * Vf0) # approx. number of atoms
            print('adding initial', Nf0, 'atoms')
            for ip in range(0, Nf0):
                fx = np.random.uniform()
                fy = np.random.uniform()
                fz = pfoil + (np.random.uniform() - 0.5) * dfoil / box[2]
                sc2.add_atom(Zf, 1.5 / (8. * np.pi**2), np.array([fx,fy,fz]))
            l_at_all = list(np.arange(0, Nf0, dtype=int))
            l_at_rem = sc2.remove_close_atoms(l_at_all, Rf*10.)
            print('removing', len(l_at_rem), 'too close atoms')
            sc2.delete_atoms(l_at_rem)
            sc3 = deepcopy(sc2)
            sc3.insert(sc1, 0) # add distance if foil and particle have crossover planes
            fcel.write_CEL(sc3, Maindir + '/cel/' + FileName + '.cel')
            #drp.cellmuncher(cel_file, Maindir + "/cel/" + FileName + "2.cif",cif = True, output = True)
            #drp.cellmuncher(Maindir + '/cel/' + FileName + '.cel', Maindir + "/cel/" + FileName + "_forcif.cif",cif = True, output = True)
        else:
            Zf = 6 # atomic number (C)
            Mf = 12.011 * ec.PHYS_MASSU # atomic mass [kg] (C) #12.011 * ec.PHYS_MASSU # atomic mass [kg] (C)# 28.0855 * ec.PHYS_MASSU # atomic mass [kg] (Si) #
            Rf = 0.160 # bond length [nm]
            rhof = 1.8 # C material density [g cm^-3] [10^-3 * 10^-21 kg * nm^-3] #2.33 # Si
            Vf0 = dfoil * box[0] * box[1] # approx. foil volume [nm^3]
            Nf0 = int(1.E-24 * rhof / Mf * Vf0) # approx. number of atoms
            print('adding initial', Nf0, 'atoms')
            for ip in range(0, Nf0):
                fx = np.random.uniform()
                fy = np.random.uniform()
                fz = pfoil + (np.random.uniform() - 0.5) * dfoil / box[2]
                sc2.add_atom(Zf, 1.5 / (8. * np.pi**2), np.array([fx,fy,fz]))
            l_at_all = list(np.arange(0, Nf0, dtype=int))
            l_at_rem = sc2.remove_close_atoms(l_at_all, Rf*10.)
            print('removing', len(l_at_rem), 'too close atoms')
            sc2.delete_atoms(l_at_rem)
            sc3 = deepcopy(sc2)
            sc3.insert(sc1, 0) # add distance if foil and particle have crossover planes
            fcel.write_CEL(sc3, Maindir + '/cel/' + FileName + '.cel')
            #drp.cellmuncher(cel_file, Maindir + "/cel/" + FileName + "2.cif",cif = True, output = True)
            #drp.cellmuncher(Maindir + '/cel/' + FileName + '.cel', Maindir + "/cel/" + FileName + "_forcif.cif",cif = True, output = True)
        
        
        drp.commands.celslc(cel_file, slice_name, ht, nx=nx, ny=ny, nz=nz, absorb=True,dwf=True, output= True)
        #msa_prm = drp.msaprm.load_msa_prm("D:/04 Code/04 Dr Probe/prm/msa.prm")
        
        # Create Parameter file for multislice
        msa = drp.msaprm.MsaPrm()
        
        # Setup (fundamental) MSA parameters
        msa.wavelength = ht  # wavelength (in nm) for 300 keV electrons
        msa.h_scan_frame_size = a 
        msa.v_scan_frame_size = b 
        msa.scan_columns = nx
        msa.scan_rows = ny
        msa.tilt_x = 1 # object tilt along x in degree
        msa.slice_files = slice_name # location of phase gratings
        msa.number_of_slices = nz # Number of slices in phase gratings
        msa.det_readout_period = 0 # Readout after every 2 slices
        msa.tot_number_of_slices = nz # Corresponds to 5 layers / unit cells of SrTiO3
        
        
        # Save prm file
        msa.save_msa_prm(Maindir + "/prm/msa.prm")
        drp.commands.msa(Maindir + "/prm/msa.prm", Maindir + "/wav/"+ "Au3" +".dat", ctem=True, output=True )
        
        # Create Parameter file for wavimg
        wavimg = drp.WavimgPrm()
        sl = nz # Perform simulation for slice # 2
        
        # Setup (fundamental) MSA parameters
        wavimg.high_tension = ht
        wavimg.mtf = (0, 0, '') 
        wavimg.oa_radius = 250 # Apply objective aperture [mrad]
        wavimg.vibration = (1, 0.025, 0.025, 0) # Apply isotropic image spread of 25 pm rms displacement
        wavimg.spat_coherence = (1, 0.4) # Apply spatial coherence
        wavimg.temp_coherence = (1, focus_spread) # Apply temp coherence, focus spread
        
        
        wavimg.wave_sampling = (a / nx, b / ny)
        #wavimg.wave_files = Maindir + "/wav/" + FileName + "_sl{:03d}.wav".format(sl)
        wavimg.wave_files = Maindir + "/wav/" + "Au3" + "_sl{:03d}.wav".format(sl)
        wavimg.wave_dim = (nx, ny)
        
        wavimg.aberrations_dict = {1: (focus, 0),2:(astx,asty)} # apply defocus of 1 nm
        #wavimg.noise # line 8
        wavimg.output_files = Maindir + "/img/" + FileName + "_sl{:03d}.dat".format(sl) # Simulate for slice number "sl"
        wavimg.output_format = 0 #output as TEM image
        wavimg.flag_spec_frame = 1 #line 9
        wavimg.output_dim = (nx>>1, ny>>1)
        wavimg.output_sampling = (a / (nx>>1))
        
        # Save wavimg Parameter file
        wavimg.save_wavimg_prm(Maindir + "/wav/wavimg.prm")
        
        drp.commands.wavimg(Maindir + "/wav/wavimg.prm", output = True)
        # Display simulated HRTEM image
        img = np.fromfile(Maindir + "/img/" + FileName + "_sl{:03d}.dat".format(sl), dtype='float32').reshape((nx>>1, ny>>1))
        #plt.imshow(img, cmap='gray');
        mpli.imsave(Maindir + "/img/" + FileName + "Si.png",img,cmap='gray')
