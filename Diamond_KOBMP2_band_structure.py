import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
from pyscf.pbc import scf  # from pyscf.pbc import gto, scf
import matplotlib.pyplot as plt

from ase.build import bulk  # Xây dựng phân tử
from ase.dft.kpoints import ibz_points, get_bandpath

from pyscf.pbc import scf as pbchf
from pyscf.pbc import df as pdf

import kobmp2_ct10_no_print_test as kobmp2  # tương tự với from pyscf.pbc import cc
import numpy as np
import sys  # Xuất file log
import time
import datetime
import os
import pandas as pd

from pyscf.pbc.df import fft

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

npoints = int(os.getenv('npoints'))
'''


kpointx = 2
kpointy = 1
kpointz = 2

star = 0
end = 2

npoints = 26


# Generate a unique filename using the current timestamp
output_filename = f"Output_band-structure_C_3D_{kpointx}x{kpointy}x{kpointz}_[{star},{end}]_KOBMP2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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

cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.verbose = 5  # Tương tự
cell.exp_to_discard = 0.1  # Loại bỏ các hàm cơ sở có exponent < 0.1, tránh tràn số (Chỗ đã chỉnh sửa)
cell.a = np.array(cell.a).tolist()  # Chuyển numpy array thành list (Chỗ đã chỉnh sửa)
cell.max_memory = 16384 # 16 GB
cell.build()  # Hơi khác 1 chút

# Tạo FFTDF object
mydf = fft.FFTDF(cell)
print(f"mydf.max_memory = {mydf.max_memory} MB")  # 6000 MB

# Có thể thay đổi sau khi tạo
mydf.max_memory = 16384  # 16 GB
print(f"Updated max_memory = {mydf.max_memory} MB")  # 16GB

# Số điểm trên đường K-path

points = ibz_points['fcc']  # Chỗ đã chỉnh sửa
G = points['Gamma']  # Chỗ đã chỉnh sửa
X = points['X']
W = points['W']
K = points['K']
L = points['L']

# band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], c.cell, npoints=npoints1) # Y chang

path = get_bandpath([L, G, X, W, K, G], cell.a , npoints=npoints)
band_kpts = path.kpts
x_axis, sp_points, labels = path.get_linear_kpoint_axis()  # Use x_axis for plotting

print("band_kpts: ")
print(band_kpts)
#band_kpts = cell.get_abs_kpts(band_kpts)
print("x_axis: ")
print(x_axis)
print("sp_points: ")
print(sp_points)

IP = np.zeros(npoints)
EA = np.zeros(npoints)

all_mo_energies = []
nocc = 0

for i in range(star, end):

    
    print("center k", band_kpts[i])
    kpts = cell.make_kpts([kpointx, kpointy, kpointz], scaled_center=band_kpts[i])
    kmf = scf.KRHF(cell, kpts, exxdiv='none')
    kmf.kernel()
    print(f"KRHF ({kpointx}x{kpointy}x{kpointz}) completed. Elapsed time: {time.time() - start_time:.2f} seconds")
    

    mypt = kobmp2.OBMP2(kmf)
    mypt.kernel()
    #print("KOBMP2 energy (per unit cell) =", mypt.e_tot)
    print(f"MP2 ({kpointx}x{kpointy}x{kpointz}) completed. Elapsed time: {time.time() - start_time:.2f} seconds")

    print("kmf.mo_energy = ", mypt.mo_energy)

    all_mo_energies.append(mypt.mo_energy[0].copy())  # Thêm dòng này

    print("all_mo_energies = ", all_mo_energies)

    IP[i] = mypt.IP
    EA[i] = mypt.EA

print(f"IP: {IP}")
print(f"EA: {EA}")


# Create filename with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = os.path.join(os.getcwd(), f"Band-structure_C_{kpointx}x{kpointy}x{kpointz}_[{star},{end}]_KOBMP2_{timestamp}.xlsx")
au2ev = 27.21139

# Tạo DataFrame với 26 hàng
df_kobmp2 = pd.DataFrame(index=range(len(x_axis)))

# Thêm cột x_axis
df_kobmp2['x_axis'] = x_axis 

mo_energy_data = np.array(all_mo_energies)  # shape (n_kpoints, n_mo)

for i in range(mo_energy_data.shape[1]):
    df_kobmp2[f'KOBMP2_{i+1}'] = np.nan

n_data_points = mo_energy_data.shape[0]  # Số điểm dữ liệu đã tính
for col_idx in range(mo_energy_data.shape[1]):
    df_kobmp2.iloc[range(star, end), col_idx+1] = mo_energy_data[:, col_idx] * au2ev    

df_kobmp2['IP'] = IP * au2ev
df_kobmp2['EA'] = EA * au2ev

# Special points
# Sửa đoạn code tạo df_sp thành:
sp_data = list(zip(sp_points, labels))
df_sp = pd.DataFrame(sp_data, columns=['x_coordinate', 'label'])  # Chỗ đã chỉnh sửa

# Conversion factor
df_au = pd.DataFrame({'au2ev': [au2ev],"nocc": [nocc]})

# Write all data to Excel file with multiple sheets
with pd.ExcelWriter(filename) as writer:
    df_kobmp2.to_excel(writer, sheet_name='KOBMP2 Bands', index=False)
    df_sp.to_excel(writer, sheet_name='Special Points', index=False)
    df_au.to_excel(writer, sheet_name='Conversion Factor', index=False)
