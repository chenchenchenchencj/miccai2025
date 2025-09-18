import nibabel as nib
import numpy as np
from skimage import measure
import pyvista as pv  # 用于3D可视化
# 1. 读取nii.gz文件
nii = nib.load("label.nii.gz")   # 你的label路径
data = nii.get_fdata()
# 2. 确保是二值mask（牙齿=1，背景=0）
mask = (data > 0).astype(np.uint8)
# 3. 使用Marching Cubes进行表面重建
verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)
# 4. 转换为pyvista的网格对象
faces = np.hstack([[3, *face] for face in faces])  # skimage返回的faces需要转换
mesh = pv.PolyData(verts, faces)
# 5. 可视化
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="white", opacity=1.0, show_edges=False)