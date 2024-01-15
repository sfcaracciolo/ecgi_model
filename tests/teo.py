import numpy as np
from open3d.geometry import TriangleMesh
from ecgi_model import EcgiModel 
from geometric_plotter import Plotter
from half_edge import HalfEdgeModel
from isotropic_remesher import IsotropicRemesher
import geometric_tools
from open3d.utility import Vector3dVector
from regularization_tools import Ridge 

outer_sphere = TriangleMesh().create_sphere(radius=1, resolution=10)
outer_model = HalfEdgeModel(outer_sphere.vertices, outer_sphere.triangles)
outer_remesher = IsotropicRemesher(outer_model)
outer_remesher.isotropic_remeshing(
    .6, # L_target, 
    iter=5, 
    explicit=False, 
    foldover=10, # degrees
    sliver=False
)
outer_remesher.model.topology_checker(clean=True)

center, radius = geometric_tools.compute_inner_sphere(outer_remesher.model.vertices)
projected_vertices = geometric_tools.project_to_sphere(outer_remesher.model.vertices, center, radius/2.)
inner_mesh = TriangleMesh(Vector3dVector(projected_vertices), outer_remesher.model.triangles)

# p = Plotter(figsize=(5.,5.))
# p.add_trisurf(np.asarray(inner_mesh.vertices), np.asarray(inner_mesh.triangles))
# p.add_trisurf(np.asarray(outer_remesher.model.vertices), np.asarray(outer_remesher.model.triangles), translate=(2,0,0))
# p.camera(view=(0,-90,0), zoom=1.)
# p.show()

# bem
_outer_data = ( np.asarray(outer_remesher.model.vertices), np.asarray(outer_remesher.model.triangles) )
_inner_data = ( np.asarray(inner_mesh.vertices), np.asarray(inner_mesh.triangles) )
ecgi_model = EcgiModel(_outer_data, _inner_data, discretization='node')

# fun
# ui0 = np.ones(outer_model.amount_of_vertices()) 
ui0 = geometric_tools.cartesian_to_spherical_coords(inner_mesh.vertices)[:, 1] # azimuth
# ue = np.ones(outer_model.amount_of_vertices()) 
ue = geometric_tools.cartesian_to_spherical_coords(outer_remesher.model.vertices)[:, 1] # azimuth

# assert np.allclose(u, ue)
ui1, uni1 = ecgi_model.solve(ue)
A, B = ecgi_model.get_transfer_matrices()
print(np.linalg.cond(A))
reg = Ridge(A)
lambdas = Ridge.lambdaspace(1e-2, 1e1, num=100)
U = reg.solve(ue, lambdas)
re = np.linalg.norm(U - ui0[:, np.newaxis], axis=1)/np.linalg.norm(ui0)
opt = np.argmin(re)

p = Plotter(figsize=(5.,5.))
p.add_trisurf(np.asarray(inner_mesh.vertices), np.asarray(inner_mesh.triangles), vertex_values=ui0, colorbar=False, vmin=ui0.min(), vmax=ui0.max(), cmap='viridis')
p.camera(view=(0,0,0), zoom=1.6)

p = Plotter(figsize=(5.,5.))
p.add_trisurf(np.asarray(inner_mesh.vertices), np.asarray(inner_mesh.triangles), vertex_values=ui1, colorbar=False, vmin=ui0.min(), vmax=ui0.max(), cmap='viridis')
p.camera(view=(0,0,0), zoom=1.6)

p = Plotter(figsize=(5.,5.))
p.add_trisurf(np.asarray(inner_mesh.vertices), np.asarray(inner_mesh.triangles), vertex_values=U[opt], colorbar=False, vmin=ui0.min(), vmax=ui0.max(), cmap='viridis')
p.camera(view=(0,0,0), zoom=1.6)

# p = Plotter(_2d=True, figsize=(5.,5.))
# p.axs.loglog(lambdas, re)

# p = Plotter(_2d=True, figsize=(5.,5.))
# p.axs.plot(ui0)
# p.axs.plot(ui1)
# # p.axs.plot(U[opt])

# p = Plotter(_2d=True, figsize=(5.,5.))
# p.axs.plot(uni1)
# p.axs.plot(B @ U[opt])
# plt.plot(uni1)
p.show()