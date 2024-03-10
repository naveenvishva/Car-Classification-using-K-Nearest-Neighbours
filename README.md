### Car Classification Model

This project contains Python code for training and evaluating a car classification model using the k-nearest neighbors algorithm. Additionally, it provides visualization of the decision boundaries using Voronoi diagrams.

### Files Included:

- `train_and_evaluate.py`: Python script for training and evaluating the KNN classifier.
- `visualization.py`: Python script for visualizing the decision boundaries using Voronoi diagrams.
- `requirements.txt`: Text file listing the required Python packages and their versions.
- `car.data`: Input dataset for training the model.
- `car_classification_model.pkl`: Trained model saved in a binary format.

### Prerequisites:

Ensure you have Python installed on your system. You can install the required packages using pip:

```
pip install -r requirements.txt
```

### Usage:

1. **Training and Evaluation**:
   - Run `train_and_evaluate.py` to train the KNN classifier on the car dataset and evaluate its performance.

2. **Visualization**:
   - After training the model, run `visualization.py` to visualize the decision boundaries using Voronoi diagrams.

### Requirements:

- pandas==1.3.3
- scikit-learn==0.24.2
- matplotlib==3.4.3
- scipy==1.7.1

### Data:
- `car.data`: Input dataset containing car attributes for training the classification model.

### Trained Model:
- `car_classification_model.pkl`: Trained KNN classifier model saved for future use.

### Note:
Ensure that you have the necessary permissions to access and load the input dataset (`car.data`) and the trained model (`car_classification_model.pkl`).

### References:
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
