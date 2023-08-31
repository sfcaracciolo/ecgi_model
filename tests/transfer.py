from matplotlib import pyplot as plt
from load import * 

"""
A way to test if the transfer matrix (A) computation is ok, it is to define two geometries with the same amount of nodes in order to get a square model.
Then, it is possible to apply gmres without the explicit A for several 'y' testing functions. 
These testing functions are computed from known solutions to compare them after solving.
If the known and estimated solutions are close, the method EcgiModel.get_transfer_matrices() is ok.
"""

fig, axs = plt.subplots(len(us), 2, sharex='col', sharey=False, width_ratios=[6, 1], figsize=(10,8))

for i, u0 in enumerate(us):
    y0 = A @ u0
    if ADD_NOISE: y0 += 1e-3*np.max(y0)*np.random.rand(y0.size)

    [u1, t], info, residuals, iterations = ecgi_model.solve(y0)

    ru = u1.coefficients - u0

    axs[i, 0].plot(u0,'-k')
    axs[i, 0].plot(ru,'--k')
    axs[i, 0].set_ylabel(ylabels[i])
    if not ADD_NOISE: axs[i, 0].set_ylim([-1,1])
    axs[i, 1].boxplot(ru, flierprops=dict(marker='.', markerfacecolor='k', markersize=5, markeredgecolor='none'))

axs[-1, 0].set_xlabel('nodes')
axs[-1, 1].set_xticks([])

plt.savefig(f"figs/transfer_{DISCRETIZATION}{'_noised' if ADD_NOISE else ''}.png", dpi = 300, orientation = 'portrait', bbox_inches = 'tight')
plt.show()