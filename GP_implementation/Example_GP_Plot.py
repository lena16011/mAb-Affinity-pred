
def plot_gpr_samples(gpr_model, n_samples, ax):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-3, 3])



#################################################################################
#### EXAMPLE PLOT GP
##############################################################################
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

n_datapoints=12
n_datapoints2=8
range=2.5
rng = np.random.RandomState(4)
# for plot with few data points
X_train = rng.uniform(0, 5, n_datapoints).reshape(-1, 1)
y_train = np.sin((X_train[:, 0]) **2 )
# for plot with more datapoints
X_train2 = X_train[X_train<2.5].reshape(-1, 1)
y_train2 = np.sin((X_train2[:, 0]) **2)
# number of sample functions to plot
n_samples = 5


# plot the true function
x = np.linspace(0, 5, 100)
X = x.reshape(-1, 1)
y = np.sin((X[:, 0]) **2 )

# generate predictions based on kernel GP
kernel = 1.0 * RBF(length_scale=1, length_scale_bounds=(1e-1, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

# set up figure
fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior 1
gpr.fit(X_train2, y_train2)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
# plot the true function
axs[1].plot(X, y, color="blue", label="True function")
# plot datapoints
axs[1].scatter(X_train2[:, 0], y_train2, color="black", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution with few data points")

fig.suptitle("Radial Basis Function kernel", fontsize=18)


# plot posterior 2
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples-5, ax=axs[2])
# plot the true function
axs[2].plot(X, y, color="blue", label="True function")
# plot datapoints
axs[2].scatter(X_train[:, 0], y_train, color="black", zorder=10, label="Observations")
#axs[2].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")    # no legend
axs[2].set_title("Samples from posterior distribution with more data points")

fig.suptitle("Radial Basis Function kernel", fontsize=18)
plt.tight_layout()

plt.show()

fig.savefig('/Users/lerlach/Documents/current_work/GP_publication/Fig3/GP_example3.pdf')
