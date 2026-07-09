import numpy as np
import pyvista as pv
import alg_utils


case_id = "POP-HT_01-08"
DIR = f"/Users/jenny/Google Drive/My Drive/POP-HT/volumetric-meshes/"
MESH_PATH = DIR + f"{case_id}/{case_id}_voxelised.alg"  # anything meshio/pyvista can read (.xdmf, .vtu, .vtk, .msh)
alg = alg_utils.read_alg_mesh(MESH_PATH)
xs, ys, zs = alg[0], alg[1], alg[2]
surface_info_field = alg[6]
trans = alg[10]
apexb = alg[12]

points = np.column_stack([xs, ys, zs]).astype(float)
cloud = pv.PolyData(points)
cloud["surface_info_field"] = surface_info_field
cloud["trans"] = trans
cloud["apexb"] = apexb

# --- 1. Full myocardium cloud, to confirm overall shape/continuity ---
p = pv.Plotter(shape=(1, 2))
p.subplot(0, 0)
p.add_mesh(cloud, scalars="trans", cmap="viridis", point_size=3, render_points_as_spheres=True)
p.add_text(f"All myocardial voxels: {cloud.n_points}\ncoloured by trans", font_size=10)

# --- 2. Endocardium only, split by LV/RV, to check for gaps/patchiness ---
endo_mask = surface_info_field != -1
endo_cloud = cloud.extract_points(endo_mask)
p.subplot(0, 1)
p.add_mesh(endo_cloud, scalars="surface_info_field", cmap="RdBu_r",
           point_size=5, render_points_as_spheres=True)
lv_n = int((surface_info_field == 0).sum())
rv_n = int((surface_info_field == 1).sum())
p.add_text(f"Endo voxels only\nLV={lv_n}, RV={rv_n}", font_size=10)
p.link_views()
p.show()

# --- 3. Quick numeric summary, useful before eyeballing ---
print(f"Total myocardial voxels: {len(xs)}")
print(f"LV endo voxels: {lv_n}, RV endo voxels: {rv_n}, non-endo: {(surface_info_field == -1).sum()}")

# Check spacing along each axis to see how sparse/gappy things actually are
for name, coord in [("x", xs), ("y", ys), ("z", zs)]:
    uniques = np.unique(coord)
    diffs = np.diff(np.sort(uniques))
    unique_diffs, counts = np.unique(diffs, return_counts=True)
    print(f"{name}: unique spacings = {dict(zip(unique_diffs.tolist(), counts.tolist()))}")