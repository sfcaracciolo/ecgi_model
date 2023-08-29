from typing import Tuple
import bempp.api
import numpy as np 

class EcgiModel:
    def __init__(self, outer_mesh: Tuple[np.ndarray, np.ndarray], inner_mesh: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        * Surfaces must have outpointing normales and be closed.
        * Surface 0 is outer Surface 1 is inner.
        """
        self.outer_mesh = outer_mesh
        self.inner_mesh = inner_mesh

        self.compute_grid()
        self.compute_spaces()
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

    def compute_spaces(self):
        """Build the BEM matrix. Implemented from [1]
        [1] Stenroos*, Matti, and Jens Haueisen. “Boundary Element Computations in the Forward and Inverse Problems of Electrocardiography: Comparison of Collocation and Galerkin Weightings.” IEEE Transactions on Biomedical Engineering 55, no. 9 (September 2008): 2124–33. https://doi.org/10.1109/TBME.2008.923913.
        """
        # http://bempp.com/handbook/api/function_spaces.html
        self.dirichlet_space_outer_surface = bempp.api.function_space(self.grid, 'P', 1, segments=[1]) 
        self.dirichlet_space_inner_surface = bempp.api.function_space(self.grid, 'P', 1, segments=[2])
        self.neumann_space_inner_surface = bempp.api.function_space(self.grid, 'P', 1, segments=[2])

    def compute_operators(self):

        self.id_i_i = bempp.api.operators.boundary.sparse.identity(
            self.dirichlet_space_inner_surface, 
            self.dirichlet_space_inner_surface,
            self.dirichlet_space_inner_surface
        )

        self.id_o_o = bempp.api.operators.boundary.sparse.identity(
            self.dirichlet_space_outer_surface, 
            self.dirichlet_space_outer_surface,
            self.dirichlet_space_outer_surface
        )

        self.v_i_i = bempp.api.operators.boundary.laplace.single_layer(
            self.neumann_space_inner_surface,
            self.dirichlet_space_inner_surface,
            self.dirichlet_space_inner_surface
        )

        self.v_i_o = bempp.api.operators.boundary.laplace.single_layer(
            self.neumann_space_inner_surface,
            self.dirichlet_space_outer_surface,
            self.dirichlet_space_outer_surface
        )

        self.k_i_i = bempp.api.operators.boundary.laplace.double_layer(
            self.dirichlet_space_inner_surface,
            self.dirichlet_space_inner_surface,
            self.dirichlet_space_inner_surface
        )

        self.k_i_o = bempp.api.operators.boundary.laplace.double_layer(
            self.dirichlet_space_inner_surface,
            self.dirichlet_space_outer_surface,
            self.dirichlet_space_outer_surface
        )

        self.k_o_o = bempp.api.operators.boundary.laplace.double_layer(
            self.dirichlet_space_outer_surface,
            self.dirichlet_space_outer_surface,
            self.dirichlet_space_outer_surface
        )

        self.k_o_i = bempp.api.operators.boundary.laplace.double_layer(
            self.dirichlet_space_outer_surface,
            self.dirichlet_space_inner_surface,
            self.dirichlet_space_inner_surface
        )

    def solve(self, values:np.ndarray):

        blocked = bempp.api.BlockedOperator(2, 2)
        blocked[0,0] = self.k_i_i - .5 * self.id_i_i
        blocked[0,1] = - self.v_i_i
        blocked[1,0] = self.k_i_o
        blocked[1,1] =  - self.v_i_o
        
        grid_fun = bempp.api.GridFunction(self.dirichlet_space_outer_surface, coefficients=values)

        rhs_fun_1 = self.k_o_i * grid_fun
        rhs_fun_2 = (self.k_o_o + .5 * self.id_o_o) * grid_fun

        return bempp.api.linalg.gmres(blocked, [rhs_fun_1, rhs_fun_2], use_strong_form=True, return_residuals=True, return_iteration_count=True)

    def get_transfer_matrices(self):
        v_i_i = self.v_i_i.strong_form().A
        v_i_o = self.v_i_o.strong_form()
        gamma = v_i_o.matmat(np.linalg.inv(v_i_i)) # ndarray
        sigma_o = (self.k_o_o + .5 * self.id_o_o).strong_form().A # ndarray
        sigma_i = (self.k_i_i - .5 * self.id_i_i).strong_form().A # ndarray
        k_o_i = self.k_o_i.strong_form().A
        b0 = gamma @ k_o_i # ndarray
        b1 = gamma @ sigma_i # ndarray
        k_i_o = self.k_i_o.strong_form()

        A = np.linalg.inv(sigma_o - b0) @ (k_i_o.A - b1)
        
        beta = k_i_o.matmat(np.linalg.inv(sigma_i))
        c0 = beta @ k_o_i
        c1 = beta @ v_i_i

        B = np.linalg.inv(sigma_o - c0) @ (c1 - v_i_o.A)
        return A, B
    