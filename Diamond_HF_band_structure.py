import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
# import pyscf.pbc.dft as pbcdft
from pyscf.pbc import scf, cc  # from pyscf.pbc import gto, scf
# import matplotlib.pyplot as plt

from ase.build import bulk  # Xây dựng phân tử
from ase.dft.kpoints import ibz_points, get_bandpath

import numpy as np
import sys  # Xuất file log
import time
import datetime
import os
import pandas as pd


from pyscf import lib
lib.num_threads(8)

# from functools import reduce

'''
Phương pháp KOBMP2
'''


class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

'''
kpointx = int(os.getenv('kpointx'))
kpointy = int(os.getenv('kpointy'))
kpointz = int(os.getenv('kpointz'))

star = int(os.getenv('star'))
end = int(os.getenv('end'))

npoints1 = int(os.getenv('npoints')) # Tăng số điểm để đường cong mượt hơn
'''

kpointx = 2
kpointy = 2
kpointz = 1

star = 0
end = 2

npoints1 = 20

# Generate a unique filename using the current timestamp
output_filename = f"output_C_{kpointx}X{kpointy}X{kpointz}_[{star},{end}]_HF_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Initialize Tee to write to both terminal and file
sys.stdout = Tee(output_filename)

start_time = time.time()

print("Starting program...")

# Step 1: Build cell
# Tạo cấu trúc bulk

c = bulk('C', 'diamond', a=3.567) # Tương tự
print(c.get_volume())

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(c)
cell.a = c.cell  # Tương tự

cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 5  # Tương tự
cell.exp_to_discard = 0.1  # Loại bỏ các hàm cơ sở có exponent < 0.1, tránh tràn số (Chỗ đã chỉnh sửa)
cell.a = np.array(cell.a).tolist()  # Chuyển numpy array thành list (Chỗ đã chỉnh sửa)
cell.build()  # Hơi khác 1 chút

# Số điểm trên đường K-path

points = ibz_points['fcc']  # Chỗ đã chỉnh sửa
G = points['Gamma']  # Chỗ đã chỉnh sửa
X = points['X']
W = points['W']
K = points['K']
L = points['L']

# band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], c.cell, npoints=npoints1) # Y chang

path = get_bandpath([L, G, X, W, K, G], cell.a , npoints=npoints1)
band_kpts = path.kpts
x_axis, sp_points, labels = path.get_linear_kpoint_axis()  # Use x_axis for plotting

print("band_kpts: ")
print(band_kpts)
#band_kpts = cell.get_abs_kpts(band_kpts)
print("x_axis: ")
print(x_axis)
print("sp_points: ")
print(sp_points)

all_mo_energies = []
nocc = 0

for i in range(star, end):

    'KRHF'
    print("center k", band_kpts[i])
    kpts = cell.make_kpts([kpointx, kpointy, kpointz], scaled_center=band_kpts[i])
    kmf = scf.KRHF(cell, kpts, exxdiv='none')
    kmf.kernel()
    print(f"KRHF ({kpointx}x{kpointy}x{kpointz}) completed. Elapsed time: {time.time() - start_time:.2f} seconds")

    print("kmf.mo_energy = ", kmf.mo_energy)

    all_mo_energies.append(kmf.mo_energy[0].copy())  # Thêm dòng này

    print("all_mo_energies = ", all_mo_energies)

    if nocc == 0:
        # Verify nocc from calculation results
        nocc = np.sum(kmf.mo_occ[0] > 0.9)
        print(f"Calculated nocc: {nocc}")
    
#au2ev = 27.21139


# Create filename with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = os.path.join(os.getcwd(), f"Band-structure_Si_{kpointx}x{kpointy}x{kpointz}_[{star},{end}]_HF_{timestamp}.xlsx")

# Create DataFrames for each section

# Tạo DataFrame cho tất cả mo_energy
au2ev = 27.21139

# Tạo DataFrame với 26 hàng
df_hf = pd.DataFrame(index=range(len(x_axis)))

# Thêm cột x_axis
df_hf['x_axis'] = x_axis 

mo_energy_data = np.array(all_mo_energies)  # shape (n_kpoints, n_mo)

# Tìm các index thỏa mãn điều kiện x_axis
#target_indices = df_hf[(df_hf['x_axis'] >= 6) & (df_hf['x_axis'] <= 8)].index
#print("target_indices", target_indices)

# Kiểm tra số lượng dữ liệu
#assert len(target_indices) == mo_energy_data.shape[0], "Số lượng k-point không khớp với dữ liệu tính toán"


for i in range(mo_energy_data.shape[1]):
    df_hf[f'hf_{i+1}'] = np.nan

n_data_points = mo_energy_data.shape[0]  # Số điểm dữ liệu đã tính
for col_idx in range(mo_energy_data.shape[1]):
    df_hf.iloc[range(star, end), col_idx+1] = mo_energy_data[:, col_idx] * au2ev    

# Special points
# Sửa đoạn code tạo df_sp thành:
sp_data = list(zip(sp_points, labels))
df_sp = pd.DataFrame(sp_data, columns=['x_coordinate', 'label'])  # Chỗ đã chỉnh sửa

# Conversion factor
df_au = pd.DataFrame({'au2ev': [au2ev], "nocc": [nocc]})  # Thêm nocc vào DataFrame

# Write all data to Excel file with multiple sheets
with pd.ExcelWriter(filename) as writer:
    df_hf.to_excel(writer, sheet_name='HF Bands', index=False)
    #df_hf.to_excel(writer, sheet_name='HF Bands', index=False)
    #df_ccsd.to_excel(writer, sheet_name='CCSD Bands', index=False)
    df_sp.to_excel(writer, sheet_name='Special Points', index=False)
    df_au.to_excel(writer, sheet_name='Conversion Factor_nocc', index=False)
