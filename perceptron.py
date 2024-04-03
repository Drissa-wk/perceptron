import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

try:
    from IPython.display import display, clear_output
except ImportError:
    IpythonNonExist = True
else:
    IpythonNonExist = False

class Perceptron:
    "Perceptron for binary classification"
    def __init__(self):
        self.W = [] #weights
        self.b = 0 #bias
        self.loss_values = [] #different values of the loss function
        self.parameters = [] #different updates values of weights and bias 
    
    def initialisation(self, X):
        "initialise weights and bias"
        w = []
        #assign random values to W
        for i in range(X.shape[1]):
            w.append(np.random.randn())
        w = np.array(w).reshape((X.shape[1], 1))
        b = 0
        return w, b
    
    def model(self, X, W, b):
        "returns activations values"
        Z = X.dot(W) + b
        A = 1 / (1 + np.exp(-Z))
        return A
    
    def loss(self, A, y):
        "Log likelihood loss function"
        T1 = y * np.log(A)
        T2 = (1-y) * np.log(1-A)
        L = (-1/len(y)) * np.sum(T1+T2)
        return L
    
    def gradient(self, X, y, A):
        "Computes the gradient of the loss function"
        dW = (1/len(y)) * np.dot(X.T, A - y) 
        db = (1/len(y)) * np.sum(A - y)
        return dW, db
    
    def train(self, X, y, learning_rate = 0.01, nb_iter = 1000):
        "train a perceptron model with data"
        
        if np.ndim(y) == 1:
            if isinstance(y, pd.Series): #If it's a pandas serie
                y = np.reshape(y.values, (len(y), 1))
            else:
                y = np.reshape(y, (len(y), 1))
            
        self.W, self.b = self.initialisation(X)
        self.parameters.append([self.W, self.b])
        
        self.loss_values = []
        for i in range(nb_iter): 
            A = self.model(X, self.W, self.b)
            self.loss_values.append(self.loss(A, y))
            dW, db = self.gradient(X, y, A)
            self.W = self.W - learning_rate * dW
            self.b = self.b - learning_rate * db
            self.parameters.append([self.W, self.b])
           
            
    def predict(self, X):
        "Use the trained model to predict values"
        Z = X.dot(self.W) + self.b
        A = 1 / (1 + np.exp(-Z))
        y_pred = A > 0.5
        return y_pred.astype(int) #vector of 0 - 1 instead of a vector of False - True
    
    def accuracy(self, y, y_pred):
        "returns the accuracy of the trained model"
        if np.ndim(y) == 1:
            if isinstance(y, pd.Series): #If it's a pandas serie
                y = np.reshape(y.values, (len(y), 1))
            else:
                y = np.reshape(y, (len(y), 1))
                
        if np.ndim(y_pred) == 1:
            y_pred = np.reshape(y_pred, (len(y), 1))

        return np.count_nonzero(y_pred == y) / len(y) #count only true values not false
     
    def draw_loss(self):
        "Draw the curve of the loss function of the trained model"
        plt.plot(self.loss_values)
    
    def draw_classification(self, X, y):
        "draw the classification line in 2D with a funny animation"
        if X.shape[1] == 2:

            #set limits of the axes
            plt.xlim(np.min(X[:, 0])-1, np.max(X[:, 0])+1)
            plt.ylim(np.min(X[:, 1])-1, np.max(X[:, 1])+1)

            if IpythonNonExist: #no animation
                plt.scatter(X[:, 0], X[:, 1], c=y)

                #plot line
                x1 = np.linspace(np.min(X[:, 0])-1, np.max(X[:, 0])+1, 100)
                x2 = (-self.W[0]*x1 - self.b) / self.W[1]
                plt.plot(x1, x2, c='r')
                plt.show()
            else:
                for i in range(0, len(self.parameters), 50):
                    plt.clf() #clear the previous fig
                    plt.scatter(X[:, 0], X[:, 1], c=y)
                    x1 = np.linspace(np.min(X[:, 0])-1, np.max(X[:, 0])+1, 100)
                    x2 = (-self.parameters[i][0][0]*x1 - self.parameters[i][1]) / self.parameters[i][0][1]
                    plt.plot(x1, x2, c='r')
                    plt.draw()
                    time.sleep(0.0004)
                    clear_output(wait=True)
                    display(plt.gcf()) #display the actual figure
                
                #plot the last line if not
                if i != len(self.parameters) - 1:
                    plt.clf() #clear the previous fig
                    plt.scatter(X[:, 0], X[:, 1], c=y)
                    x1 = np.linspace(np.min(X[:, 0])-1, np.max(X[:, 0])+1, 100)
                    x2 = (-self.parameters[i][0][0]*x1 - self.parameters[i][1]) / self.parameters[i][0][1]
                    plt.plot(x1, x2, c='r')
                    plt.draw()
                    time.sleep(0.0004)
                    clear_output(wait=True)
                    display(plt.gcf()) #display the actual figure

                plt.close()
        else:
            print("We can only draw graph in 2D !")
        
    
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
    y = np.reshape(y, (len(y), 1))
    perceptron = Perceptron()
    perceptron.train(X, y)
    y_pred = perceptron.predict(X)
    print(perceptron.accuracy(y, y_pred))
    # perceptron.draw_loss()
    perceptron.draw_classification(X, y)