import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import joblib
from scipy.spatial import Voronoi, voronoi_plot_2d

# Load the trained model
model = joblib.load('car_classification_model.pkl')

# Create a meshgrid for visualization
xx, yy = np.meshgrid(np.arange(-1, 5, 0.01), np.arange(-1, 5, 0.01))
mesh_predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
mesh_predictions = mesh_predictions.reshape(xx.shape)

# Create a Voronoi diagram
vor = Voronoi(model._fit_X)

# Plot the Voronoi diagram and decision boundaries
fig, ax = plt.subplots(figsize=(8, 6))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=2, line_alpha=0.6)
plt.contourf(xx, yy, mesh_predictions, cmap=plt.cm.coolwarm, alpha=0.3)
plt.scatter(model._fit_X[:, 0], model._fit_X[:, 1], c=model._y, cmap='Set1')
plt.title('Voronoi Diagram with Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
