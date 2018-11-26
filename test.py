import numpy as np
from LogisticRegression import  *
from statistics import  mean
iris = load_iris()
x = np.matrix([[1],[2],[3]])
t = np.matrix([[2],[3],[6]])
print( np.multiply(x,t))
X = np.matrix(iris.data[:,:2])
Y = (iris.target!=0)*1
Y = Y.reshape(Y.shape[0],1)
lr = 0.1
num_iter = 1000
model = LogisticRegression()
model.fit(X, Y)
#CPU times: user 13.8 s, sys: 84 ms, total: 13.9 s
#Wall time: 13.8 s
preds = model.predict(X)
# accuracy
(preds == Y).mean()
1.0

