# ECGI (ElectroCardioGraphic Imaging) Model

A BEM (Boundary Element Method) solver of Laplace problem with Cauchy boundary conditions: 

$$ \nabla^2 \phi(\mathbf{r}) = 0 \quad \mathbf{r}\in \Omega $$

$$ \phi(\mathbf{r}) = \phi_e \quad \mathbf{r}\in \partial\Omega_e $$

$$ \frac{\partial \phi}{\partial \mathbf{n}}(\mathbf{r}) = 0 \quad \mathbf{r}\in \partial\Omega_e $$

where the domain is a hollow volumen like this:
<img src="figs/domain.png" alt="drawing" width="347"/>

the model require two meshes as input. The inner and outer surfaces, both must be closed and outpointing normal surfaces.

The implementation depends on [bempp](https://bempp.com/) and  has a `solve()` method that require a square system, i.e., inner and outer meshes with same amount of nodes. Otherwise, `get_transfer_matrices()` is prefered in order to apply some regularization because the problem is ill-posed.

## Usage

```python
from ecgi_model import EcgiModel

outer_mesh = (
    outer_vertices, # numpy.ndarray [No x 3],
    outer_triangles, # numpy.ndarray [Mo x 3],
)

inner_mesh = (
    inner_vertices, # numpy.ndarray [Ni x 3],
    inner_triangles, # numpy.ndarray [Mi x 3],
)

ecgi_model = EcgiModel(outer_mesh, inner_mesh)
A, B = model.get_transfer_matrices()
```