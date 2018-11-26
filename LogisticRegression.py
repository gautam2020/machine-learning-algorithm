import numpy as np
import sklearn as sk
from sklearn.datasets import load_iris
from statistics import  mean

class LogisticRegression:
    def __int__(self):
        self.lr= 0.01
        self.num_iter = 100
        self.fit_intercept = True
        self.verbose = True

    def _add_intercept(self,X):
        intercept = np.ones((X.shape[0],1))
        return np.concatenate((intercept,X),axis=1)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def loss(self,h,y):
        print("here")
        return (-np.multiply(y,np.log(h))-np.multiply((1-y),np.log(1-h))).mean()

    def fit(self,X,y):
        X = self._add_intercept(X)
        print(X.shape[1])
        self.theta = np.zeros((X.shape[1],1))
        for i in range(1000000):
          z = np.dot(X, self.theta)
          h = self.sigmoid(z)
          gradient = (np.dot(X.T,(h-y)))/y.size
          self.theta = self.theta - 0.1*gradient
          if(i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
                print(f'loss: {self.loss(h, y)} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold



