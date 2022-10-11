import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm

### new imports
import warnings
from sklearn.cluster import KMeans
from scipy.stats import norm
###


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0


### new global variables
NUM_CLUSTERS = 20
SEED = 0
SCALE_FEATURES = 1
DECISION_SHIFT_SCALER = norm.ppf(COST_W_UNDERPREDICT / (COST_W_UNDERPREDICT + COST_W_OVERPREDICT))


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary
        self.gps = [None for _ in range(NUM_CLUSTERS)]
        self.kmeans = None
        ### TODO: try more
        ### https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html (ConvergenceWarning)
        # self.kernels = [ConstantKernel() * Matern() + Matern(),
        #                 ConstantKernel() * RBF(length_scale_bounds=(1e-5, 1)) + RBF(length_scale_bounds=(1e-5, 10))]
        self.kernels = [1.0 * RBF(length_scale=1e-2, length_scale_bounds=(1e-3, 1e3)) +
                        WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-10, 1e1))]

    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean = np.zeros(test_features.shape[0], dtype=float)
        gp_std = np.zeros(test_features.shape[0], dtype=float)

        # # TODO: Use the GP posterior to form your predictions here
        # predictions = gp_mean

        assert self.kmeans is not None
        cluster_pred = self.kmeans.predict(test_features)  # (N,)
        all_clusters_pred = np.unique(cluster_pred)
        for cluster_iter in all_clusters_pred:
            mask_iter = (cluster_pred == cluster_iter)
            test_features_iter = test_features[mask_iter]
            gp_iter = self.gps[cluster_iter]
            gp_mean[mask_iter], gp_std[mask_iter] = gp_iter.predict(test_features_iter, return_std=True)

        predictions = gp_mean + DECISION_SHIFT_SCALER * gp_std

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_GT: np.ndarray, train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        # print(train_GT.shape)  # (15189,)
        train_features *= SCALE_FEATURES
        self._partition_domain_by_k_means(train_features)
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore")
            for cluster_ind in range(NUM_CLUSTERS):
                if_print = cluster_ind % 5 == 0
                self._fit_one_gp(train_GT, train_features, cluster_ind)
                if if_print:
                    print(f"current: {cluster_ind + 1}/{NUM_CLUSTERS}")
                    # print(self.gps[cluster_ind])

    def _partition_domain_by_k_means(self, train_features: np.ndarray):
        # train_GT: (N, 2)
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED).fit(train_features)
        self.kmeans = kmeans

    def _fit_one_gp(self, train_GT: np.ndarray, train_features: np.ndarray, cluster_ind):
        best_score = -float("inf")
        best_gp = None
        select_mask = (self.kmeans.labels_ == cluster_ind)
        train_GT = train_GT[select_mask]
        train_features = train_features[select_mask]
        for kernel_iter in self.kernels:
            gpr = GaussianProcessRegressor(kernel=kernel_iter.clone_with_theta(kernel_iter.theta),
                                           random_state=SEED).fit(train_features, train_GT)
            log_likelihood = gpr.log_marginal_likelihood()
            if best_score < log_likelihood:
                best_score = log_likelihood
                best_gp = gpr

        self.gps[cluster_ind] = best_gp


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT,train_features)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
