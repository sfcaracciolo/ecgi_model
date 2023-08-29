from half_edge import HalfEdgeModel
from isotropic_remesher import IsotropicRemesher
from open3d.utility import Vector3dVector
from open3d.geometry import TriangleMesh
import open3d as o3d 
import pathlib
import numpy as np 
from ecgi_model import EcgiModel 
import geometric_tools 
import geometric_plotter
import matplotlib
import matplotlib.pyplot as plt 

path = pathlib.Path('examples/outer_mesh.npz')
if not path.exists(): 
    outer_sphere = TriangleMesh().create_sphere(radius=1, resolution=10)
    outer_model = HalfEdgeModel(outer_sphere.vertices, outer_sphere.triangles)
    outer_remesher = IsotropicRemesher(outer_model)
    outer_remesher.isotropic_remeshing(
        .2, # L_target, 
        iter=5, 
        explicit=False, 
        foldover=10, # degrees
        sliver=False
    )

    outer_remesher.model.topology_checker(clean=True)
    np.savez(path, vertices=outer_model.vertices, triangles=outer_model.triangles)
else:
    npz = np.load(path)
    outer_model = HalfEdgeModel(npz['vertices'], npz['triangles'])

outer_mesh = TriangleMesh(outer_model.vertices, outer_model.triangles)

center, radius = geometric_tools.compute_inner_sphere(outer_model.vertices)
projected_vertices = geometric_tools.project_to_sphere(outer_model.vertices, center, radius/2.)
inner_mesh = TriangleMesh(Vector3dVector(projected_vertices), outer_model.triangles)

# bem
_outer_data = ( np.asarray(outer_model.vertices), np.asarray(outer_model.triangles) )
_inner_data = ( np.asarray(inner_mesh.vertices), np.asarray(inner_mesh.triangles) )
ecgi_model = EcgiModel(_outer_data, _inner_data)

# spherical = geometric_tools.cartesian_to_spherical_coords(outer_model.vertices)
# ρ, θ, φ = spherical[:,0], spherical[:,1], spherical[:,2]
# values = np.cos(φ) 
values = np.ones(outer_model.amount_of_vertices())

[u, t], info, residuals, iterations = ecgi_model.solve(values)

vmin, vmax = 0, 2
# vmin, vmax = min(values.min(), u.coefficients.min()), max(values.max(), u.coefficients.max())
print(vmin, vmax)
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap('viridis')
outer_mesh.vertex_colors = Vector3dVector(cmap(norm(values))[:,:-1])
inner_mesh.vertex_colors = Vector3dVector(cmap(norm(u.coefficients))[:,:-1])

A, B = ecgi_model.get_transfer_matrices()
print(np.allclose(A @ u.coefficients, values, rtol=1e-3))
# print(np.allclose(B @ t.coefficients, values, rtol=1e-1))
# print(B @ t.coefficients)

o3d.visualization.draw_geometries([inner_mesh.translate([2,0,0]), outer_mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

# geometric_plotter.set_export()



# vmin, vmax = min(values.min(), u.coefficients.min()), max(values.max(), u.coefficients.max())
# print(vmin, vmax)

# ax = geometric_plotter.figure(figsize=(5,5))

# geometric_plotter.plot_trisurf(ax, *_outer_data, vertex_values=values, vmin=-3, vmax=3)

# _inner_data[0] += np.array([[2,0,0]]) # translate
# geometric_plotter.plot_trisurf(ax, *_inner_data, vertex_values=u.coefficients, vmin=-3, vmax=3)

# geometric_plotter.config_ax(ax, (50,50,0), 1.5)

# geometric_plotter.execute(folder='E:\Repositorios\isotropic_outer_remesher\export\\', name='square')
