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

from pyscf.pbc.df import fft

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
kpointy = 1
kpointz = 1

star = 22
end = 24

npoints1 = 26

# Generate a unique filename using the current timestamp
output_filename = f"output_C_{kpointx}X{kpointy}X{kpointz}_[{star},{end}]_CCSD_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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

cell.basis = 'gth-tzvp'
cell.pseudo = 'gth-pade'
cell.verbose = 5  # Tương tự
cell.exp_to_discard = 0.1  # Loại bỏ các hàm cơ sở có exponent < 0.1, tránh tràn số (Chỗ đã chỉnh sửa)
cell.a = np.array(cell.a).tolist()  # Chuyển numpy array thành list (Chỗ đã chỉnh sửa)
cell.max_memory = 16384 # 16 GB
cell.build()  # Hơi khác 1 chút

mydf = fft.FFTDF(cell)
mydf.max_memory = cell.max_memory  # 16 GB 
print(f"FFTDF max_memory = {mydf.max_memory} MB")

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

list_ip = []  # HOMO-1 (second highest occupied)
list_ea = []  # LUMO+1 (second lowest unoccupied)

for i in range(star, end):

    'KRHF'
    print("center k", band_kpts[i])
    kpts = cell.make_kpts([kpointx, kpointy, kpointz], scaled_center=band_kpts[i])
    kmf = scf.KRHF(cell, kpts, exxdiv='none')
    kmf.max_memory = mydf.max_memory
    kmf.kernel()
    print(f"KRHF ({kpointx}x{kpointy}x{kpointz}) completed. Elapsed time: {time.time() - start_time:.2f} seconds")

    print("kmf.mo_energy = ", kmf.mo_energy)

    'KRCCSD'
    mycc = cc.KRCCSD(kmf)
    mycc.kernel()

    # CẢI TIẾN 1: Xác định số roots tối đa cho IP/EA
    nocc = mycc.nocc  # Số orbital occupied
    nmo = mycc.nmo    # Tổng số orbital
    max_ip_roots = nocc
    max_ea_roots = nmo - nocc
    print(f"max_ip_roots = {max_ip_roots}, max_ea_roots = {max_ea_roots}")

    # CẢI TIẾN 2: Tính toán tất cả roots tại k=0
    ip = mycc.ipccsd(nroots=max_ip_roots, kptlist=[0])  # Tất cả IP roots
    ea = mycc.eaccsd(nroots=max_ea_roots, kptlist=[0])  # Tất cả EA roots
    print(f"IP roots: {ip[0]}")
    print(f"EA roots: {ea[0]}")
    print(f"IP SIZE: {len(ip[0])}, EA SIZE: {len(ea[0])}")
    # CẢI TIẾN 3: Lưu trữ động theo số roots

    list_ip.append([-x for x in ip[0][0].tolist()])  # Thêm cả list IP roots
    list_ea.append([x for x in ea[0][0].tolist()])
    

print(f"list_ip: {list_ip}")
print(f"list_ea: {list_ea}")


au2ev = 27.21139

# Create filename with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = os.path.join(os.getcwd(), f"Band-structure_C_{kpointx}x{kpointy}x{kpointz}_HF_CCSD_{timestamp}.xlsx")

# Tạo DataFrame với 26 hàng
df_CCSD = pd.DataFrame(index=range(len(x_axis)))

# Thêm cột x_axis
df_CCSD['x_axis'] = x_axis 

list_ip_data = np.array(list_ip)  # shape (n_kpoints, n_mo)
list_ea_data = np.array(list_ea)  # shape (n_kpoints, n_mo)

for i in range(list_ip_data.shape[1]):
    df_CCSD[f'CCSD_ip_{i+1}'] = np.nan

for i in range(list_ea_data.shape[1]):
    df_CCSD[f'CCSD_ea_{i+1}'] = np.nan

n_data_points1 = list_ip_data.shape[0]  # Số điểm dữ liệu đã tính
n_data_points1 = list_ea_data.shape[0]  # Số điểm dữ liệu đã tính

for col_idx in range(list_ip_data.shape[1]):
    df_CCSD.iloc[range(star, end), col_idx+1] = list_ip_data[:, col_idx] * au2ev   

# Ghi dữ liệu EA vào các cột tiếp theo (sau các cột IP)
for col_idx in range(list_ea_data.shape[1]):
    df_CCSD.iloc[range(star, end), col_idx+1+list_ip_data.shape[1]] = list_ea_data[:, col_idx] * au2ev

#######
# Special points
# Sửa đoạn code tạo df_sp thành:
sp_data = list(zip(sp_points, labels))
df_sp = pd.DataFrame(sp_data, columns=['x_coordinate', 'label'])  # Chỗ đã chỉnh sửa

# Conversion factor
df_au = pd.DataFrame({'au2ev': [au2ev]})

# Write all data to Excel file with multiple sheets
with pd.ExcelWriter(filename) as writer:
    #df_hf.to_excel(writer, sheet_name='HF Bands', index=False)
    df_CCSD.to_excel(writer, sheet_name='CCSD Bands', index=False)
    df_sp.to_excel(writer, sheet_name='Special Points', index=False)
    df_au.to_excel(writer, sheet_name='Conversion Factor', index=False)
