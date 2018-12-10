import os
import numpy as np
import glob
import pdb
import math

def find_bin(n, edges):
    if n < edges[0] or n > edges[-1]:
        return np.nan
    else:
        return np.where(n > edges)[0][-1]

def electrode_mapping(electrodes, xedges=None, yedges=None):
    if yedges is None:
        yedges = np.array([0., 8., 16., 24., 32.,   40.,  48., 56.,  64.,  72.,  80.])
    if xedges is None:
        xedges = np.array([-30., -22., -14., -6.,  2.,   10.,   18.,  26.,  34.,  42.,  50.])
    #electrodes = np.vstack([np.array([float(x) for x in channel.split(",")]) for channel in mni_file])
    mapping = {}
    for c in range(64):
        new_c = find_bin(electrodes[c, 2], yedges) * len(xedges) + find_bin(electrodes[c, 1], xedges)
        if not np.isnan(new_c):
            mapping[c] = new_c
    return mapping


def convert_ellipsoidalCoords2MNI(angleVals, center, radii):
    angleVals = np.deg2rad(angleVals)  # convert angles to radians
    x = radii[0] * np.sin(angleVals[:, 0]) * np.cos(angleVals[:, 1]) + center[0]
    y = radii[1] * np.sin(angleVals[:, 0]) * np.sin(angleVals[:, 1]) + center[1]
    z = radii[2] * np.cos(angleVals[:, 0]) + center[2]

    # Switch values for true locations (originally put z along MNI y-axis for less warping at top of the head)
    x1 = np.copy(y)
    y1 = np.copy(z)
    z1 = np.copy(x)

    outcoords = np.array([x1, y1, z1])
    return outcoords

def print_virtual_electrodes():
    # Adapted from Steven Peterson's code

    # 1: Create virtual grid

    # Estimated ellipsoid parameters
    corticalCenter = np.array([20.8971, 0, -16.4751])  # MNI coordinates (but z,x,y)
    corticalRadii = np.array([59.7079, 65.6566, 77.2230])  # MNI coordinates (but z,x,y)

    # Create virtual grid points
    numGridPts = 400  # create 20x20 grid for now
    N = int(math.sqrt(numGridPts))
    phiVals = np.zeros((N, N))
    thetaVals = np.zeros((N, N))
    for i in range(N):
        phiVals[i, :] = np.linspace(10, 105, N)  # max/min phi angles
        thetaVals[:, i] = np.linspace(-130, 130, N)  # max/min theta angles

    # Downsample to appropriate grid size
    phiVals2 = np.transpose(phiVals[:int(N / 2), :])
    thetaVals2 = np.transpose(thetaVals[:, 0:int(N / 2)])
    angleVals = np.transpose(np.array([phiVals2.flatten(), thetaVals2.flatten()]))

    # Convert ellipsoidal to MNI coordinates
    virtualCoords = convert_ellipsoidalCoords2MNI(angleVals, corticalCenter, corticalRadii)

    # Only keep coordinates with x<0 (single hemisphere)
    virtualCoords_pos = virtualCoords[:, virtualCoords[0, :] < 0]

    return virtualCoords_pos

def VirtualGrid_ellipsoid_mapping(realCoords):
    # Adapted from Steven Peterson's code

    # Project realCoords to x<0:
    realCoords[:,0] = -np.abs(realCoords[:,0])

    #1: Create virtual grid

    #Estimated ellipsoid parameters
    corticalCenter=np.array([20.8971, 0, -16.4751]) #MNI coordinates (but z,x,y)
    corticalRadii=np.array([59.7079, 65.6566, 77.2230]) #MNI coordinates (but z,x,y)

    #Create virtual grid points
    numGridPts=400 #create 20x20 grid for now
    N=int(math.sqrt(numGridPts))
    phiVals=np.zeros((N,N))
    thetaVals=np.zeros((N,N))
    for i in range(N):
        phiVals[i,:]=np.linspace(10,105,N) #max/min phi angles
        thetaVals[:,i]=np.linspace(-130,130,N) #max/min theta angles

    #Downsample to appropriate grid size
    phiVals2=np.transpose(phiVals[:int(N/2),:])
    thetaVals2=np.transpose(thetaVals[:,0:int(N/2)])
    angleVals=np.transpose(np.array([phiVals2.flatten(),thetaVals2.flatten()]))

    #Convert ellipsoidal to MNI coordinates
    virtualCoords=convert_ellipsoidalCoords2MNI(angleVals,corticalCenter,corticalRadii)

    #Only keep coordinates with x<0 (single hemisphere)
    virtualCoords_pos=virtualCoords[:,virtualCoords[0,:]<0]

    #2: find closest virtual grid coordinate for all grid electrodes

    #assuming realCoords is (n x 3) in size

    N_real=realCoords.shape[0]
    mapping={}
    xmax = virtualCoords_pos[0].max()
    xmin = virtualCoords_pos[0].min()
    ymax = virtualCoords_pos[1].max()
    ymin = virtualCoords_pos[1].min()
    zmax = virtualCoords_pos[2].max()
    zmin = virtualCoords_pos[2].min()

    for i in range(N_real):
        if (realCoords[i,0] < xmax+5) & (realCoords[i,0] > xmin-5) & \
                (realCoords[i,1] < ymax+5) & (realCoords[i,1] > ymin-5) & \
                (realCoords[i,2] < zmax+5) & (realCoords[i,2] > zmin-5):
            distances=np.sqrt(np.square(realCoords[i,0]-virtualCoords_pos[0,:])+np.square(realCoords[i,1]-virtualCoords_pos[1,:])+ \
                              np.square(realCoords[i,2]-virtualCoords_pos[2,:]))

            mapping[i]=np.argmin(distances)
    return mapping


def main():
    subjects = ['294e1c','69da36', 'ecb43e', 'c5a5e9']
    subject_id_map = {'69da36': 'd65', '294e1c': 'a0f', 'c5a5e9': 'c95', 'ecb43e': 'cb4'}
    mni_dir = '/home/wangnxr/Documents/mni_coords/'
    main_data_dir = "/data2/users/wangnxr/dataset/"
    for subject in subjects:
	
        print "Working on subject %s" % subject
        mni_file = np.loadtxt("%s/%s_Trodes_MNIcoords.txt" % (mni_dir, subject), delimiter=",")
        mapping=VirtualGrid_ellipsoid_mapping(mni_file)
	for file in glob.glob("%s/ecog_vid_combined_%s_day*/*/*/*.npy" % (main_data_dir, subject_id_map[subject])):
            print file
	    if not os.path.exists("%s/ecog_mni_ellipv2_%s/%s" % (main_data_dir, subject_id_map[subject], "/".join(file.split("/")[-3:]))):
		print "skip"
		continue
            orig = np.load(file)
            new = np.zeros(shape=(100, orig.shape[1]))
            for old_c, new_c in mapping.iteritems():
                if old_c < 64:
                    new[new_c] = orig[old_c]
            file_parts = file.split("/")
            try:
                np.save("%s/ecog_mni_ellip_%s/%s" % (main_data_dir, subject_id_map[subject], "/".join(file_parts[-3:])), new)
            except IOError:
                os.makedirs("%s/ecog_mni_ellip_%s/%s" % (main_data_dir, subject_id_map[subject], "/".join(file_parts[-3:-1])))
                np.save(
                    "%s/ecog_mni_ellip_%s/%s" % (main_data_dir, subject_id_map[subject], "/".join(file_parts[-3:])),
                    new)

if __name__=='__main__':
    main()

