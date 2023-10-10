from typing import Literal, Tuple
import bempp.api
import numpy as np 

class EcgiModel:
    def __init__(self, outer_mesh: Tuple[np.ndarray, np.ndarray], inner_mesh: Tuple[np.ndarray, np.ndarray], discretization: Literal['node', 'triangle'] = 'node') -> None:
        """
        * Surfaces must have outpointing normales and be closed.
        * Surface 1 is outer Surface 2 is inner.
        """
        self.outer_mesh = outer_mesh
        self.inner_mesh = inner_mesh

        if discretization == 'node':
            poly = ('P', 1)
        elif discretization == 'triangle':
            poly = ('DP', 0)
        else:
            raise ValueError('Discretization does not exist.')
        
        self.compute_grid()
        self.compute_spaces(poly)
        self.compute_operators()

    def compute_grid(self):
        outer_vertices, outer_triangles = self.outer_mesh
        outer_V, outer_T = outer_vertices.shape[0], outer_triangles.shape[0]

        inner_vertices, inner_triangles = self.inner_mesh
        inner_V, inner_T = inner_vertices.shape[0], inner_triangles.shape[0]

        # greometry compound
        vertices = np.vstack((outer_vertices, inner_vertices))
        triangles = np.vstack((outer_triangles, inner_triangles + outer_V))
        T = outer_T + inner_T

        # domain ids: 1 outer, 2 inner
        ids = np.ones(T, dtype=np.uint32)
        ids[outer_T:] = 2
        self.grid = bempp.api.Grid(vertices.T, triangles.T, domain_indices=ids)

    def compute_spaces(self, poly):
        self.outer_space = bempp.api.function_space(self.grid, *poly , segments=[1]) 
        self.inner_space = bempp.api.function_space(self.grid, *poly , segments=[2])

    def compute_operators(self):

        self.id_i_i = bempp.api.operators.boundary.sparse.identity(
            self.inner_space, 
            self.inner_space,
            self.inner_space
        )

        self.id_o_o = bempp.api.operators.boundary.sparse.identity(
            self.outer_space, 
            self.outer_space,
            self.outer_space
        )

        self.v_i_i = bempp.api.operators.boundary.laplace.single_layer(
            self.inner_space,
            self.inner_space,
            self.inner_space
        )

        self.v_i_o = bempp.api.operators.boundary.laplace.single_layer(
            self.inner_space,
            self.outer_space,
            self.outer_space
        )

        self.k_i_i = bempp.api.operators.boundary.laplace.double_layer(
            self.inner_space,
            self.inner_space,
            self.inner_space
        )

        self.k_i_o = bempp.api.operators.boundary.laplace.double_layer(
            self.inner_space,
            self.outer_space,
            self.outer_space
        )

        self.k_o_o = bempp.api.operators.boundary.laplace.double_layer(
            self.outer_space,
            self.outer_space,
            self.outer_space
        )

        self.k_o_i = bempp.api.operators.boundary.laplace.double_layer(
            self.outer_space,
            self.inner_space,
            self.inner_space
        )

        self.u_i = self.k_i_i - .5 * self.id_i_i
        self.u_o = self.k_o_o + .5 * self.id_o_o

    def solve(self, values:np.ndarray, return_all : bool = False):

        if values.ndim > 1:
            values = np.ravel(values)

        blocked = bempp.api.BlockedOperator(2, 2)
        blocked[0,0] = self.u_i
        blocked[0,1] = - self.v_i_i
        blocked[1,0] = self.k_i_o
        blocked[1,1] =  - self.v_i_o
        
        grid_fun = bempp.api.GridFunction(self.outer_space, coefficients=values)

        rhs_fun_1 = self.k_o_i * grid_fun
        rhs_fun_2 = self.u_o * grid_fun

        (u, un), res, _iter, info = bempp.api.linalg.gmres(blocked, [rhs_fun_1, rhs_fun_2], use_strong_form=True, return_residuals=True, return_iteration_count=True)

        if return_all:
            return (u, un), res, _iter, info
        
        return u.coefficients, un.coefficients

    def get_transfer_matrices(self):

        V_i_i = self.v_i_i.strong_form().A
        V_i_o = self.v_i_o.strong_form().A
        U_o = self.u_o.strong_form().A 
        U_i = self.u_i.strong_form().A 
        K_o_i = self.k_o_i.strong_form().A
        K_i_o = self.k_i_o.strong_form().A

        W = V_i_o @ np.linalg.inv(V_i_i) 
        A = np.linalg.inv(U_o - W @ K_o_i) @ (K_i_o - W @ U_i)
        
        Z = K_o_i @ np.linalg.inv(U_o) 
        B = np.linalg.inv(V_i_i - Z @ V_i_o) @ (U_i - Z @ K_i_o)

        return A, B
    