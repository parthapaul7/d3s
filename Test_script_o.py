from D3S import D3S
import os
import argparse

parser = argparse.ArgumentParser(description="Compute D3S(BJ) and D3(BJ) energies from scan .xyz file")
parser.add_argument("xyz_file", type=str, help="Path to the scan.xyz file")
args = parser.parse_args()

# Extract the scan file path and base name
xyz_path = args.xyz_file


S6,S8,a = D3S.DFT_D3ZERO_parameters["BLYP"]
COORDS = D3S.read_XYZ(xyz_path)

#compute D3S(BJ) energies for the scan geometries
D3S_BJ_energy_list = []
for COORD in COORDS:
    D3S_BJ_energy_list.append(D3S.D3SZERO(COORD, S6,S8,a))

#compute D3(BJ) energies for the scan geometries
D3_BJ_energy_list = []
k3 = 4
for COORD in COORDS:
    D3_BJ_energy_list.append(D3S.D3ZERO(COORD, S6,S8,a,k3))

with open("D3S_BJ_scan.txt", 'w') as f:
    for line in D3S_BJ_energy_list:
        f.write(f"{line}\n")

with open("D3_BJ_scan.txt", 'w') as f:
    for line in D3_BJ_energy_list:
        f.write(f"{line}\n")