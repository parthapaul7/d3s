import numpy as np
import os
from copy import deepcopy
import pickle

# Get the directory of the D3S.py
current_dir = os.path.dirname(os.path.abspath(__file__))

#Import R_cov table
r_cov_path = os.path.join(current_dir, "R_cov.txt")
R_cov = np.genfromtxt(r_cov_path, delimiter=",")
R_cov = R_cov[~np.isnan(R_cov)]  
#Import R4/R2 table
r4r2_path = os.path.join(current_dir, "R2R4.txt")
R4R2 = np.genfromtxt(r4r2_path, delimiter=",") #sqrt(0.5*r2r4(i)*dfloat(i)**0.5 )
R4R2 = R4R2[~np.isnan(R4R2)] 
#Import R0_AB cutoff distances table
r0_ab_path = os.path.join(current_dir, "R0_ab.txt")
R0_AB = np.genfromtxt(r0_ab_path, delimiter=",")
R0_AB = R0_AB[~np.isnan(R0_AB)]  
#Transform flat R0_AB into 2D-array with Z => i-1. I.e. hydrogen is 0.
R0_AB_2d = np.zeros((94,94))
inds = list(zip(*[(i,j-1) for j in range(1,95) for i in range(j)]))
inds = (np.array(inds[0]), np.array(inds[1]))
R0_AB_2d[inds] = R0_AB 
i_lower = np.tril_indices(94, -1)
R0_AB_2d[i_lower] = R0_AB_2d.T[i_lower]  # make the matrix symmetric
#Import C6AB table
c6ab_path = os.path.join(current_dir, "Params_C6.txt")
C6AB_raw = np.genfromtxt(c6ab_path, delimiter=",")
C6AB_raw = C6AB_raw[~np.isnan(C6AB_raw)] 
#Transform flat C6AB into 5D-array: C6AB[IAT, JAT, IADR, JADR, M] where IAT, JAT - element numbers; 
#                                   IADR, JADR - number of reference; M - 0=C6 coef, 1=CN_A, 2=CN_b.
def LIMIT(IAT,JAT):
    IADR = 0
    JADR = 0
    while IAT>100:
        IAT=IAT-100
        IADR=IADR+1
    while JAT>100:
        JAT=JAT-100
        JADR=JADR+1
    return IAT-1, JAT-1, IADR, JADR
NLINES = 32385
M = 3
L = 5
K = 5
J = 94
I = 94
C6AB = -np.ones([I,J,K,L,M])
MAXCI = np.zeros(J)
KK = 0
for NN in range(NLINES):
    IAT=int(C6AB_raw[KK+1])
    JAT=int(C6AB_raw[KK+2])
    IAT, JAT, IADR, JADR = LIMIT(IAT, JAT)
    MAXCI[IAT] = max(MAXCI[IAT],IADR)
    MAXCI[JAT] = max(MAXCI[JAT],JADR)

    C6AB[IAT,JAT,IADR,JADR,0]=C6AB_raw[KK]
    C6AB[IAT,JAT,IADR,JADR,1]=C6AB_raw[KK+3]
    C6AB[IAT,JAT,IADR,JADR,2]=C6AB_raw[KK+4]
    C6AB[JAT,IAT,JADR,IADR,0]=C6AB_raw[KK]
    C6AB[JAT,IAT,JADR,IADR,1]=C6AB_raw[KK+4]
    C6AB[JAT,IAT,JADR,IADR,2]=C6AB_raw[KK+3]
    KK=((NN+1)*5) 
MAXCI = np.array(MAXCI, dtype=np.int64)
#Import atomic-pair dependent k3 parameters
k3_2d_path = os.path.join(current_dir, "K3_2D_list")
with open(k3_2d_path, 'rb') as f:
    K3_2D_list = pickle.load(f)

Periodic_Table = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
    "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118}

#Functions to compute D3S correction (BJ and zero damping are implemented so far)
AUTOANG = 0.52917726

def L(CN_I, CN_J, ref_CN_I, ref_CN_J,k3):
    return np.exp(-k3*((CN_I-ref_CN_I)**2+(CN_J-ref_CN_J)**2))

def Z(I,J,CN_I,CN_J,k3):
    res = 0
    for n in range(MAXCI[I]+1):
        for k in range(MAXCI[J]+1):
            res += C6AB[I][J][n][k][0]*L(CN_I,CN_J,C6AB[I][J][n][k][1],C6AB[I][J][n][k][2],k3)
    return res

def W(I,J,CN_I,CN_J,k3):
    res = 0
    for n in range(MAXCI[I]+1):
        for k in range(MAXCI[J]+1):
            res += L(CN_I,CN_J,C6AB[I][J][n][k][1],C6AB[I][J][n][k][2],k3)
    return res

def L_k3_PD(CN_I, CN_J, ref_CN_I, ref_CN_J, I, J):
    k3 = K3_2D_list[I][J]
    return np.exp(-k3*((CN_I-ref_CN_I)**2+(CN_J-ref_CN_J)**2))

def Z_k3_PD(I,J,CN_I,CN_J):
    res = 0
    for n in range(MAXCI[I]+1):
        for k in range(MAXCI[J]+1):
            res += C6AB[I][J][n][k][0]*L_k3_PD(CN_I,CN_J,C6AB[I][J][n][k][1],C6AB[I][J][n][k][2], I, J)
    return res

def W_k3_PD(I,J,CN_I,CN_J):
    res = 0
    for n in range(MAXCI[I]+1):
        for k in range(MAXCI[J]+1):
            res += L_k3_PD(CN_I,CN_J,C6AB[I][J][n][k][1],C6AB[I][J][n][k][2], I, J)
    return res

def C6(I,J,CN_I,CN_J,k3):
    return Z(I,J,CN_I,CN_J,k3)/W(I,J,CN_I,CN_J,k3)

def C6_k3_PD(I,J,CN_I,CN_J):
    return Z_k3_PD(I,J,CN_I,CN_J)/W_k3_PD(I,J,CN_I,CN_J)

def CN(ATOM_IND, COORDS):
    target = COORDS[ATOM_IND]
    res = 0
    for d,atoms in enumerate(COORDS):
        if d!=ATOM_IND:
            R_ij = np.linalg.norm(target[1]-atoms[1])
            res += 1/(1+np.exp(-16*((R_cov[target[0]-1]+R_cov[atoms[0]-1])*(4/3)/R_ij-1)))
    return res

def C8(C6_AB,I,J):
    return 3*C6_AB*R4R2[I]*R4R2[J]

def BJ(ATOM_1, ATOM_2, ALPHA_1, ALPHA_2, C6, C8, N):
    I, J = (ATOM_1[0]-1), (ATOM_1[0]-1)
    R_IJ_cutoff = np.sqrt(C8/C6)
    R_IJ = np.linalg.norm(ATOM_1[1]-ATOM_2[1])/AUTOANG
    return R_IJ**N/(R_IJ**N + (ALPHA_1*R_IJ_cutoff + ALPHA_2)**N)

def ZERO(ATOM_1, ATOM_2, ALPHA_N, SR_N):
    I, J = (ATOM_1[0]-1), (ATOM_2[0]-1)
    R_IJ_cutoff = R0_AB_2d[I][J]
    R_IJ = np.linalg.norm(ATOM_1[1]-ATOM_2[1])
    return (1 + 6*(R_IJ/(SR_N*R_IJ_cutoff))**(-ALPHA_N))**(-1)

#We left the flexibolity to tune k3 parameter for standart D3 model. The default value is k3=4.
def D3BJ(COORDS, S6, S8, ALPHA_1, ALPHA_2, k3):
    Dispersion_energy_E6 = 0
    Dispersion_energy_E8 = 0
    CN_array = []
    for ATOM_IND in range(len(COORDS)):
        CN_array.append(CN(ATOM_IND, COORDS))
    for d1, ATOM_1 in enumerate(COORDS[:-1]):
        for d2, ATOM_2 in enumerate(COORDS[d1+1:]):
            R_IJ = np.linalg.norm(ATOM_1[1]-ATOM_2[1])/AUTOANG #Divide by the transformation from A to borh
            I, J = (ATOM_1[0]-1), (ATOM_2[0]-1)
            C6_AB = C6(I,J,CN_array[d1],CN_array[d2+d1+1],k3)
            C8_AB = C8(C6_AB,I,J)
            Dispersion_energy_E6 += S6*BJ(ATOM_1, ATOM_2, ALPHA_1, ALPHA_2, C6_AB, C8_AB, 6)*C6_AB/R_IJ**6
            Dispersion_energy_E8 += S8*BJ(ATOM_1, ATOM_2, ALPHA_1, ALPHA_2, C6_AB, C8_AB, 8)*C8(C6_AB,I,J)/R_IJ**8
    return -(Dispersion_energy_E8+Dispersion_energy_E6)#, Dispersion_energy_E6, Dispersion_energy_E8, Dispersion_energy_E8/(Dispersion_energy_E8+Dispersion_energy_E6) 

def D3SBJ(COORDS, S6, S8, ALPHA_1, ALPHA_2):
    Dispersion_energy_E6 = 0
    Dispersion_energy_E8 = 0
    CN_array = []
    for ATOM_IND in range(len(COORDS)):
        CN_array.append(CN(ATOM_IND, COORDS))
    for d1, ATOM_1 in enumerate(COORDS[:-1]):
        for d2, ATOM_2 in enumerate(COORDS[d1+1:]):
            R_IJ = np.linalg.norm(ATOM_1[1]-ATOM_2[1])/AUTOANG #Divide by the transformation from A to borh
            I, J = (ATOM_1[0]-1), (ATOM_2[0]-1)
            C6_AB = C6_k3_PD(I,J,CN_array[d1],CN_array[d2+d1+1])
            C8_AB = C8(C6_AB,I,J)
            Dispersion_energy_E6 += S6*BJ(ATOM_1, ATOM_2, ALPHA_1, ALPHA_2, C6_AB, C8_AB, 6)*C6_AB/R_IJ**6
            Dispersion_energy_E8 += S8*BJ(ATOM_1, ATOM_2, ALPHA_1, ALPHA_2, C6_AB, C8_AB, 8)*C8(C6_AB,I,J)/R_IJ**8
    return -(Dispersion_energy_E8+Dispersion_energy_E6)#, Dispersion_energy_E6, Dispersion_energy_E8, Dispersion_energy_E8/(Dispersion_energy_E8+Dispersion_energy_E6) 


def D3ZERO(COORDS, S6, S8, SR6, k3):
    Dispersion_energy_E6 = 0
    Dispersion_energy_E8 = 0
    CN_array = []
    for ATOM_IND in range(len(COORDS)):
        CN_array.append(CN(ATOM_IND, COORDS))
    for d1, ATOM_1 in enumerate(COORDS[:-1]):
        for d2, ATOM_2 in enumerate(COORDS[d1+1:]):
            R_IJ = np.linalg.norm(ATOM_1[1]-ATOM_2[1])/AUTOANG #Divide by the transformation from A to borh
            I, J = (ATOM_1[0]-1), (ATOM_2[0]-1)
            C6_AB = C6(I,J,CN_array[d1],CN_array[d2+d1+1],k3)
            C8_AB = C8(C6_AB,I,J)
            Dispersion_energy_E6 += S6*ZERO(ATOM_1, ATOM_2, 14, SR6)*C6_AB/R_IJ**6
            Dispersion_energy_E8 += S8*ZERO(ATOM_1, ATOM_2, 16, 1)*C8(C6_AB,I,J)/R_IJ**8
    return -(Dispersion_energy_E8+Dispersion_energy_E6)#, Dispersion_energy_E6, Dispersion_energy_E8, Dispersion_energy_E8/(Dispersion_energy_E8+Dispersion_energy_E6) 


def D3SZERO(COORDS, S6, S8, SR6):
    Dispersion_energy_E6 = 0
    Dispersion_energy_E8 = 0
    CN_array = []
    for ATOM_IND in range(len(COORDS)):
        CN_array.append(CN(ATOM_IND, COORDS))
    for d1, ATOM_1 in enumerate(COORDS[:-1]):
        for d2, ATOM_2 in enumerate(COORDS[d1+1:]):
            R_IJ = np.linalg.norm(ATOM_1[1]-ATOM_2[1])/AUTOANG #Divide by the transformation from A to borh
            I, J = (ATOM_1[0]-1), (ATOM_2[0]-1)
            C6_AB = C6_k3_PD(I,J,CN_array[d1],CN_array[d2+d1+1])
            C8_AB = C8(C6_AB,I,J)
            Dispersion_energy_E6 += S6*ZERO(ATOM_1, ATOM_2, 14, SR6)*C6_AB/R_IJ**6
            Dispersion_energy_E8 += S8*ZERO(ATOM_1, ATOM_2, 16, 1)*C8(C6_AB,I,J)/R_IJ**8
    return -(Dispersion_energy_E8+Dispersion_energy_E6)#, Dispersion_energy_E6, Dispersion_energy_E8, Dispersion_energy_E8/(Dispersion_energy_E8+Dispersion_energy_E6) 


def read_XYZ(file_name):
    with open(file_name, 'r') as f:
        file = f.read().split('\n')
    clean_file = []
    for line in file:
        if line != '>' and line != '':
            clean_file.append(line)
    LEN = len(clean_file)
    NUM_STRUCT = LEN//(int(clean_file[0])+2)
    COORDS = []
    for i in range(NUM_STRUCT):
        structure = []
        for atoms in range(int(clean_file[0])):
            ind, x, y, z = clean_file[atoms+2+i*(int(clean_file[0])+2)].split()
            ind = Periodic_Table[ind]
            x, y, z = float(x), float(y), float(z)
            structure.append([ind, np.array([x,y,z])])
        COORDS.append(structure)
    return COORDS

#Import D3BJ optimizad parameters for different functionals
DFT_D3BJ_parameters = {}
d3bj_param_path = os.path.join(current_dir, "DFT_D3BJ_parameters.txt")
with open(d3bj_param_path,'r') as f:
    file = f.read()
    file = file.split('\n')
    for line in file:
        name, s6, s8, a1, a2 = line.split()
        s6 = float(s6)
        s8 = float(s8)
        a1 = float(a1)
        a2 = float(a2)
        DFT_D3BJ_parameters[name] = (s6,s8,a1,a2)

#Import D3(0) optimizad parameters for different functionals
DFT_D3ZERO_parameters = {}
d30_param_path = os.path.join(current_dir, "DFT_D30_parameters.txt")
with open(d30_param_path,'r') as f:
    file = f.read()
    file = file.split('\n')
    for line in file:
        name, s6, s8, sr6 = line.split()
        s6 = float(s6)
        s8 = float(s8)
        sr6 = float(sr6)
        DFT_D3ZERO_parameters[name] = (s6,s8,sr6)