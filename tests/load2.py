from half_edge import HalfEdgeModel
from isotropic_remesher import IsotropicRemesher
from open3d.utility import Vector3dVector
from open3d.geometry import TriangleMesh
import pathlib
from ecgi_model import EcgiModel 
import geometric_tools 
import numpy as np 

ADD_NOISE = False
DISCRETIZATION = 'node'

outer_sphere = TriangleMesh().create_sphere(radius=1, resolution=10)
outer_mesh = TriangleMesh(outer_sphere.vertices, outer_sphere.triangles)

inner_sphere = TriangleMesh().create_sphere(radius=.5, resolution=10)
inner_mesh = TriangleMesh(inner_sphere.vertices, inner_sphere.triangles)

# bem
_outer_data = ( np.asarray(outer_mesh.vertices), np.asarray(outer_mesh.triangles) )
_inner_data = ( np.asarray(inner_mesh.vertices), np.asarray(inner_mesh.triangles) )
ecgi_model = EcgiModel(_outer_data, _inner_data, discretization=DISCRETIZATION)
A, B = ecgi_model.get_transfer_matrices()

# smooth functions
spherical = geometric_tools.cartesian_to_spherical_coords(inner_mesh.vertices)
ρ, θ, φ = spherical[:,0], spherical[:,1], spherical[:,2]
us = (  np.full(φ.size, .5), np.cos(φ), np.cos(θ), np.cos(φ)*np.sin(θ) )
if DISCRETIZATION == 'triangle': 
    f = lambda x: geometric_tools.interp_vertices_values_to_triangles(inner_mesh.vertices, inner_mesh.triangles, x)
    us = list(map(f, us))

ylabels = [
    '$1/2$',
    '$\cos(φ)$',
    '$\cos(θ)$',
    '$\cos(φ)\sin(θ)$',
]

