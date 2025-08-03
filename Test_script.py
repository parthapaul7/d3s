from D3S import D3S
import os
import argparse

parser = argparse.ArgumentParser(description="Compute D3S(BJ) and D3(BJ) energies from scan .xyz file")
parser.add_argument("xyz_file", type=str, help="Path to the scan.xyz file")
args = parser.parse_args()

# Extract the scan file path and base name
xyz_path = args.xyz_file


S6,S8,a,b = D3S.DFT_D3BJ_parameters["BLYP"]
COORDS = D3S.read_XYZ(xyz_path)

print(len(COORDS[0]))

#compute D3S(BJ) energies for the scan geometries
D3S_BJ_energy_list = []
for COORD in COORDS:
    D3S_BJ_energy_list.append(D3S.D3SBJ(COORD, S6,S8,a,b))

CN_list_cat = []
for COORD in COORDS:
    CN_list_cat.append(D3S.CN(0,COORD))

CN_list_an = []
for COORD in COORDS:
    CN_list_an.append(D3S.CN(len(COORD[0])-1,COORD))


#compute D3(BJ) energies for the scan geometries
D3_BJ_energy_list = []
k3 = 4
for COORD in COORDS:
    D3_BJ_energy_list.append(D3S.D3BJ(COORD, S6,S8,a,b,k3))

with open("D3S_BJ_scan.txt", 'w') as f:
    for line in D3S_BJ_energy_list:
        f.write(f"{line}\n")

with open("D3_BJ_scan.txt", 'w') as f:
    for line in D3_BJ_energy_list:
        f.write(f"{line}\n")

with open("CN_cat_scan.txt", 'w') as f:
    for line in CN_list_cat:
        f.write(f"{line}\n")

with open("CN_an_scan.txt", 'w') as f:
    for line in CN_list_an:
        f.write(f"{line}\n")