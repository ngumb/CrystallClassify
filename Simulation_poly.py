
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
Build a large gold fcc structure object (size of the supercell) and then 
proceed to take cutouts from thet structure

"""



ImgNum =  10000 # Number of images to simulate

Maindir = "C:/Au-NPs" # Location to store the simulation files and subfolders
MonoGold = "C:/MonoGold.txt" # Location of the MonoGold.txt file


ht = 300
nx = 512
ny = 512
nz = 64
a = 6.0621
b = a

for j in range(1,ImgNum):
    '''random parameters, (rotation, size, focus, astigmatism)'''
    alpha = np.random.uniform()*360
    beta = np.random.uniform()*360
    gamma = np.random.uniform()*360
    focus = 1+(2*np.random.uniform()-1)*5
    astx = (2*np.random.uniform()-1)*3
    asty = (2*np.random.uniform()-1)*3
    focus_spread = 5+(2*np.random.uniform()-1)*0.5
    NPsize =  1+np.random.uniform()*3
    numCuts = np.random.randint(1,high = 3) # To Do: TEST!!!
    
    RotM=R.from_euler("XYZ", [alpha,beta,gamma])
   
    FileName = 'MonoGold' + '_Poly' +'_numCuts{:.0f}'.format(numCuts) +  '_j{:.0f}'.format(j)
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
    
    
    # read in coordinates from original file
    A= pd.read_csv(MonoGold, sep = '\s+')
    xyz = A[['x','y','z']]
    
    # Remove all entries that are out of bounds
    NPra = NPsize/2/a
    BoxMin = 0.5-NPra
    BoxMax = 0.5+NPra
    xyz["distance"] = ((0.5 - xyz['x'])**2+(0.5 - xyz['y'])**2+(0.5 - xyz['z'])**2)**0.5
    XYZ = xyz[xyz['distance'] < NPra]
    xyz = XYZ[['x','y','z']]
    
   
    xyz=xyz*10*a;
    # Get Centre and Shift
    xyzShift = np.array(xyz.min()-BoxD)
    xyzMid = np.array(xyz.min()+(xyz.max()-xyz.min())/2).reshape([1,3])
    # Apply rotation
    xyz = xyz-xyzMid
    
    ''' POLY PROCEDURE 
    Make two or more separate Particles (randomly initialize)
    Rotate each one with a different rotation matrix
    Randomly slice and stick together'''
    
    xyzRot = RotM.apply(xyz)
    
    for cut in range(numCuts):
        alpha = np.random.uniform()*180
        beta = np.random.uniform()*180
        gamma = np.random.uniform()*180
        RotM=R.from_euler("XYZ", [alpha,beta,gamma])
        
        xyzRot2 = RotM.apply(xyz) #rotated Part1
        
        xcut = (2*np.random.uniform()-1) * (NPsize/2*0.5)
        
        xyzRot = pd.DataFrame(xyzRot, columns = ['x','y','z']).copy()
        xyzRot2 = pd.DataFrame(xyzRot2, columns = ['x','y','z']).copy()
        # cut xyzRot
        xyzRot = xyzRot[xyzRot['x'] < xcut ]
        #cut xyzRot2
        xyzRot2 = xyzRot2[xyzRot2['x'] >= xcut ]
       
        xyzRot = pd.concat([xyzRot,xyzRot2])
        xyzRot = np.array(xyzRot)
        alpha = np.random.uniform()*180
        beta = np.random.uniform()*180
        gamma = np.random.uniform()*180
        RotM=R.from_euler("XYZ", [alpha,beta,gamma])
        xyzRot = RotM.apply(xyzRot) #rotate Rot
    
    
    xyz = xyzRot + xyzMid
    
    
    # write coordinates to cel file
    for i in range(xyz.shape[0]):
        f = (xyz[i,:]-xyzShift)/10/a
        sc1.add_atom(Zp, uiso, f)

    
    # make a supercell with a foil of randomly placed atoms
    sc2 = supercell()
    sc2.a0 = box.copy() * 10.
    # Foil position parameters
    dfoil = 1+np.random.uniform()*2 # foil thickness [nm]
    pfoil = ((xyz[:,2].max()-xyzShift[2])/10+dfoil/2)/a # foil center position (fract)
    
    if j<=ImgNum/2:
        Zf = 14 # atomic number (Si)
        Mf = 28.0855 * ec.PHYS_MASSU # atomic mass [kg] (Si) #12.011 * ec.PHYS_MASSU # atomic mass [kg] (C)
        Rf = 0.160 # bond length [nm]
        rhof = 2.33 # Si 1.8 # C material density [g cm^-3] [10^-3 * 10^-21 kg * nm^-3] 
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
    msa.save_msa_prm(Maindir + "/prm/msa3.prm")
    drp.commands.msa(Maindir + "/prm/msa3.prm", Maindir + "/wav/"+ "Au" +".dat", ctem=True, output=True )
    
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
    wavimg.wave_files = Maindir + "/wav/" + "Au" + "_sl{:03d}.wav".format(sl)
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
    mpli.imsave(Maindir + "/img/" + FileName + "_Si.png",img,cmap='gray')
