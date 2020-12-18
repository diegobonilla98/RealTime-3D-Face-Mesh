import open3d
import numpy as np

pcd = open3d.io.read_point_cloud('landmarks.txt', format='xyz')

R = pcd.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
pcd.rotate(R, center=(0, 0, 0))

pcd.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(10)

# pcd.colors = open3d.utility.Vector3dVector(colors)
with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
    rec_mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
rec_mesh.paint_uniform_color([0.3, 0.3, 0.3])
rec_mesh.compute_vertex_normals()
open3d.visualization.draw_geometries([pcd, rec_mesh], mesh_show_back_face=True)

pcd = rec_mesh.sample_points_poisson_disk(6000)
pcd.normals = open3d.utility.Vector3dVector(np.zeros((1, 3)))
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(10)
open3d.visualization.draw_geometries([pcd], point_show_normal=True)

with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
    rec_mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
rec_mesh.compute_vertex_normals()
open3d.visualization.draw_geometries([rec_mesh], mesh_show_back_face=True)
