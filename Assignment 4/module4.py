
from multilayer_perceptron import MultilayerPerceptron
import numpy as np

_weight1=np.array([[.1,.2,.3],[.4,.5,.6]])
_bias1=np.array([.1,.2,.1])
_weight2=np.array([[.1,.3],[.4,.6],[.2,.4]])
_bias2=np.array([.1,.1])
_weight3=np.array([[.2,.3],[.5,.6]])
_bias3=np.array([.1,.2])
mlp=MultilayerPerceptron(_weight1, _bias1, _weight2, _bias2, _weight3, _bias3)
x=np.array([2,3])
y=mlp.forward(x)
print(x,y)


_weight1=np.random.rand(2,3)
_bias1=np.random.rand(3,)
_weight2=np.random.rand(3,2)
_bias2=np.random.rand(2,)
_weight3=np.random.rand(2,2)
_bias3=np.random.rand(2,)
mlp=MultilayerPerceptron(_weight1, _bias1, _weight2, _bias2, _weight3, _bias3)
x=np.array([2,3])
y=mlp.forward(x)
print(x,y)