from half_edge import HalfEdgeModel
from isotropic_remesher import IsotropicRemesher
from open3d.geometry import TriangleMesh
import open3d as o3d 
import pathlib
import numpy as np 
from ecgi_model import EcgiModel 
from open3d.utility import Vector3dVector
import geometric_tools 
import matplotlib
import matplotlib.pyplot as plt 

inner_path = pathlib.Path('inner_mesh.npz')
if not inner_path.exists(): 
    inner_sphere = TriangleMesh().create_sphere(radius=1, resolution=10)
    inner_model = HalfEdgeModel(inner_sphere.vertices, inner_sphere.triangles)
    inner_remesher = IsotropicRemesher(inner_model)
    inner_remesher.isotropic_remeshing(
        .4, # L_target, 
        iter=5, 
        explicit=False, 
        foldover=10, # degrees
        sliver=False
    )

    inner_remesher.model.topology_checker(clean=True)
    np.savez(inner_path, vertices=inner_model.vertices, triangles=inner_model.triangles)
else:
    npz = np.load(inner_path)
    inner_model = HalfEdgeModel(npz['vertices'], npz['triangles'])

inner_mesh = TriangleMesh(inner_model.vertices, inner_model.triangles)


outer_path = pathlib.Path('outer_mesh.npz')
if not outer_path.exists(): 
    outer_sphere = TriangleMesh().create_sphere(radius=2, resolution=15)
    outer_model = HalfEdgeModel(outer_sphere.vertices, outer_sphere.triangles)
    outer_remesher = IsotropicRemesher(outer_model)
    outer_remesher.isotropic_remeshing(
        .4, # L_target, 
        iter=5, 
        explicit=False, 
        foldover=10, # degrees
        sliver=False
    )
    outer_remesher.model.topology_checker(clean=True)
    np.savez(outer_path, vertices=outer_model.vertices, triangles=outer_model.triangles)
else:
    npz = np.load(outer_path)
    outer_model = HalfEdgeModel(npz['vertices'], npz['triangles'])

outer_mesh = TriangleMesh(outer_model.vertices, outer_model.triangles)


# bem
_outer_data = (np.asarray(outer_mesh.vertices), np.asarray(outer_mesh.triangles))
_inner_data = (np.asarray(inner_mesh.vertices), np.asarray(inner_mesh.triangles))

ecgi_model = EcgiModel(_outer_data, _inner_data)
# A, B = model.get_transfer_matrices()
spherical = geometric_tools.cartesian_to_spherical_coords(outer_model.vertices)
ρ, θ, φ = spherical[:,0], spherical[:,1], spherical[:,2]
values = np.cos(φ) 

[u, t], info, residuals, iterations = ecgi_model.solve(values)
print(info, iterations)


norm = matplotlib.colors.Normalize(vmin=-3, vmax=3)
cmap = plt.get_cmap('viridis')
outer_mesh.vertex_colors = Vector3dVector(cmap(norm(values))[:,:-1])
inner_mesh.vertex_colors = Vector3dVector(cmap(norm(u.coefficients))[:,:-1])

A, B = ecgi_model.get_transfer_matrices()
print(np.allclose(A @ u.coefficients, values, rtol=1e-3))
print(np.allclose(B @ t.coefficients, values, rtol=1e-1))

o3d.visualization.draw_geometries([inner_mesh.translate([2,0,0]), outer_mesh], mesh_show_wireframe=True, mesh_show_back_face=True)