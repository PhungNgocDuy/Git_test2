import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pandas as pd  # Thêm thư viện pandas
import numpy

from ase.build import bulk
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath

#c = bulk('Si', 'diamond', a=5.431) # Tương tự
#c = bulk('Ge', 'diamond', a=5.658)
c = bulk('C', 'diamond', a=3.5668)
print(c.get_volume())

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(c)
cell.a = c.cell

cell.basis = 'gth-tzvp'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.exp_to_discard = 0.1  # Loại bỏ các hàm cơ sở có exponent < 0.1, tránh tràn số (Chỗ đã chỉnh sửa)
cell.a = numpy.array(cell.a).tolist() 
#cell.build(None, None)
cell.build()

points = special_points['fcc']
G = points['G']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], c.cell, npoints=110)
band_kpts = cell.get_abs_kpts(band_kpts)

#
# band structure from Gamma point sampling
#
"""
mf = pbcdft.RKS(cell)
print(mf.kernel())

e_kn = mf.get_bands(band_kpts)[0]
vbmax = -99
for en in e_kn:
    vb_k = en[cell.nelectron // 2 - 1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn = [en - vbmax for en in e_kn]
"""
#
# band structure from 222 k-point sampling
#

kmf = pbcdft.KRKS(cell, cell.make_kpts([3, 3, 3]))
kmf.xc = 'PBE'
print(kmf.kernel())

e_kn_2 = kmf.get_bands(band_kpts)[0]
vbmax = -99
for en in e_kn_2:
    vb_k = en[cell.nelectron // 2 - 1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn_2 = [en - vbmax for en in e_kn_2]

au2ev = 27.21139  # Hệ số chuyển đổi Hartree → eV

# Tạo DataFrame chứa kết quả
nbands = cell.nao_nr()
band_data = []

# Thu thập dữ liệu cho từng k-point
for i, k in enumerate(kpath):
    row = {'kpath': k}
    for band_idx in range(nbands):
        #row[f'band_{band_idx}_gamma'] = e_kn[i][band_idx] * au2ev
        row[f'band_{band_idx}_222'] = e_kn_2[i][band_idx] * au2ev
    band_data.append(row)

# Tạo DataFrame từ dữ liệu
df_bands = pd.DataFrame(band_data)

# Tạo DataFrame cho các điểm đặc biệt
df_special_points = pd.DataFrame({
    'label': ['L', 'Gamma', 'X', 'W', 'K', 'Gamma'],
    'position': sp_points
})

# Lưu kết quả vào file Excel
with pd.ExcelWriter('[QT]band_structure_results_C_110_3x3x3_PBE_TZVP.xlsx') as writer:
    df_bands.to_excel(writer, sheet_name='Band Energies', index=False)
    df_special_points.to_excel(writer, sheet_name='Special Points', index=False)

print("Kết quả đã được lưu vào file band_structure_results.xlsx")



