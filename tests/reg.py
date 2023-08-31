from regularization_tools import Regularizer
from matplotlib import pyplot as plt
from load import * 
"""
Similar to transfer.py, but instead gmres a regularization is performed.
"""

model = Regularizer.ridge(A)
m, M = .01*model.lambda_range[0], 100.*model.lambda_range[1]
lambdas = model.lambda_logspace(m, M, 100)
model.compute_filter_factors(lambdas)

fig, axs = plt.subplots(len(us), 2, sharex='col', sharey=False, width_ratios=[6, 1], figsize=(10,8))

for i, u0 in enumerate(us):
    u0 = u0[:,np.newaxis]
    y0 = A @ u0
    if ADD_NOISE: y0 += 1e-3*np.max(y0)*np.random.rand(y0.size)[:,np.newaxis]

    U1 = model.solve(y0)
    rus = np.abs(U1 - u0)
    ix_opt = np.median(rus, axis=1).argmin()
    u1 = U1[ix_opt]
    ru = u1 - u0

    axs[i, 0].plot(u0,'-k')
    axs[i, 0].plot(ru,'--k')
    axs[i, 0].set_ylabel(ylabels[i])
    if not ADD_NOISE: axs[i, 0].set_ylim([-1,1])
    axs[i, 1].boxplot(ru, flierprops=dict(marker='.', markerfacecolor='k', markersize=5, markeredgecolor='none'))

axs[-1, 0].set_xlabel('nodes')
axs[-1, 1].set_xticks([])

plt.savefig(f"figs/reg_{DISCRETIZATION}{'_noised' if ADD_NOISE else ''}.png", dpi = 300, orientation = 'portrait', bbox_inches = 'tight')
plt.show()