import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os
matplotlib.use('Agg')
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                  [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1) with shift
    X2 = np.random.multivariate_normal(mean=[1 + distance, 1 + distance], cov=covariance_matrix, size=n_samples)
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []

    n_samples = step_num
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(20, n_rows * 10))

    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
        
        # Fit the model and record parameters
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)

        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)
        
        # Calculate logistic loss
        loss = -np.mean(y * np.log(model.predict_proba(X)[:, 1]) + (1 - y) * np.log(1 - model.predict_proba(X)[:, 1]))
        loss_list.append(loss)

        # Plot the dataset
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')

        # Plot decision boundary and margin contours
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Plot fading contours
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
        
        # Calculate margin width for 70% confidence contours
        class_1_contour = plt.contour(xx, yy, Z, levels=[0.7], colors='red', linestyles='--')
        class_0_contour = plt.contour(xx, yy, Z, levels=[0.3], colors='blue', linestyles='--')
        class_1_vertices = class_1_contour.collections[0].get_paths()[0].vertices
        class_0_vertices = class_0_contour.collections[0].get_paths()[0].vertices
        min_distance = np.min(cdist(class_1_vertices, class_0_vertices))
        margin_widths.append(min_distance)

        plt.title(f"Shift Distance = {distance}", fontsize=16)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.text(x_min + 0.1, y_max - 1.0, f"{beta0:.2f} + {beta1:.2f}*x1 + {beta2:.2f}*x2 = 0", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    # Plot parameter changes with shift distances
    plt.figure(figsize=(18, 15))

    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, label="Beta0")
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, label="Beta1")
    plt.title("Shift Distance vs Beta1")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, label="Beta2")
    plt.title("Shift Distance vs Beta2")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, label="Slope (Beta1 / Beta2)")
    plt.title("Shift Distance vs Slope")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, label="Intercept (Beta0 / Beta2)")
    plt.title("Shift Distance vs Intercept")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, label="Logistic Loss")
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, label="Margin Width")
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
