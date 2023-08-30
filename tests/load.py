from half_edge import HalfEdgeModel
from isotropic_remesher import IsotropicRemesher
from open3d.utility import Vector3dVector
from open3d.geometry import TriangleMesh
import pathlib
from ecgi_model import EcgiModel 
import geometric_tools 
import numpy as np 

path = pathlib.Path('geo/outer_mesh.npz')
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
_outer_data = ( np.asarray(outer_mesh.vertices), np.asarray(outer_mesh.triangles) )
_inner_data = ( np.asarray(inner_mesh.vertices), np.asarray(inner_mesh.triangles) )
ecgi_model = EcgiModel(_outer_data, _inner_data)

path = pathlib.Path('tests/mat.npz')
if not path.exists(): 
    A, B = ecgi_model.get_transfer_matrices()
    np.savez(path, A=A, B=B)
else:
    npz = np.load(path)
    A, B = npz['A'], npz['B']

# smooth functions
spherical = geometric_tools.cartesian_to_spherical_coords(outer_model.vertices)
ρ, θ, φ = spherical[:,0], spherical[:,1], spherical[:,2]
us = (  np.full(φ.size, .5), np.cos(φ), np.cos(θ), np.cos(φ)*np.sin(θ) )
ylabels = [
    '$1/2$',
    '$\cos(φ)$',
    '$\cos(θ)$',
    '$\cos(φ)\sin(θ)$',
]