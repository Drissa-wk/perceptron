
# Perceptron

Unlocking Binary Classification Potential: Harnessing the Power of Perceptron for Efficient Training, Accurate Prediction, and Insightful Visualization

## I. Using the Perceptron class

To use the Perceptron class, follow these steps:

### 1. Import the Perceptron class

```python
from perceptron import Perceptron
```

### 2. Initialize a Perceptron object:

```python
perceptron = Perceptron()
```

Train the model with your data:

```python
perceptron.train(X, y, learning_rate=0.01, nb_iter=1000)
```

* **X**: the training data
* **y**: the corresponding class labels
* **learning_rate**: the learning rate (default = 0.01)
* **nb_iter**: the number of training iterations (default = 1000)

### 3. Make predictions with the trained model:

```python
y_pred = perceptron.predict(X_test)
```

### 4. Evaluate the accuracy of the model:

```python
accuracy = perceptron.accuracy(y_test, y_pred)
```

### 5. Display the loss function curve:

```python
perceptron.draw_loss()
```

### 6. Visualize the classification in a 2D plot:

```python
perceptron.draw_classification(X, y)
```

> **Note :** You can use the notebook **perceptron.ipynb**, which already contains all of this procedure and uses real world data to make classifications.

## II. Contributions and Modifications
If you wish to make modifications to the Perceptron class or contribute to the project, please follow these guidelines:

### 1. Clone the repository:

```bash
git clone https://github.com/Drissa-wk/perceptron.git
```

### 2. Make modifications

Modify the perceptron code as per your requirements.
Ensure to document the modifications made.

### 3. Propose modifications

* Create a branch for your modifications
 ```bash
git checkout -b your_branch_name
```

* Add and commit your modifications
```bash
git commit -m "Description of your modifications"
```

* Push your modifications to the remote repository
```bash
git push origin your_branch_name
```

### 4. Create a Pull Request on GitHub to propose your modifications

We encourage you to contribute to the improvement of this Perceptron class. 

Thank you for your participation!

