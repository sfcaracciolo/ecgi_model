from half_edge import HalfEdgeModel
from isotropic_remesher import IsotropicRemesher
from open3d.utility import Vector3dVector
from open3d.geometry import TriangleMesh
import open3d as o3d 
import pathlib
import numpy as np 
from ecgi_model import EcgiModel 
import geometric_tools 
from regularization_tools import Regularizer

"""
Similar to transfer.py, but instead gmres a regularization
is performed. 'u' and 'y' errores enhance as expected.
"""
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
us = (  np.cos(φ)**0, np.cos(φ)**2, np.cos(φ)**4, np.cos(φ)**6, np.cos(φ)**8 )

model = Regularizer.ridge(A)
m, M = .1*model.lambda_range[0], 10.*model.lambda_range[1]
lambdas = model.lambda_logspace(m, M, 100)
model.compute_filter_factors(lambdas)

for i, u0 in enumerate(us):
    u0 = u0[:,np.newaxis]
    y0 = A @ u0
    # y0 += 0.01*np.mean(y0)*np.random.rand(y0.size)
    U1 = model.solve(y0)
    rus = np.abs(U1 - u0)/np.abs(u0)

    ix_opt = np.mean(rus, axis=1).argmin()
    ru = rus[:, ix_opt]

    y1 = A @ U1[ix_opt]
    ry = np.abs(y1 - y0)/np.abs(y0) # if y is smooth, this error is low.

    print(f'{i+1}/{len(us)}')
    print(f'u errors: mean = {np.mean(ru):.2}, max = {np.max(ru):.2})')
    print(f'y errors: mean = {np.mean(ry):.2}, max = {np.max(ry):.2})')
    assert np.allclose(y0, y1) # , rtol=1e-4)

print('OK')