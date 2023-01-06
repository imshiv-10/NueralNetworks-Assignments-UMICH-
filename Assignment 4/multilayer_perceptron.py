import numpy as np
class MultilayerPerceptron:
    def __init__(self,_weight1,_bias1,_weight2,_bias2,_weight3,_bias3):
        self.net={}
        self.net['_weight1']=_weight1
        self.net['_bias1']=_bias1
        self.net['_weight2']=_weight2
        self.net['_bias2']=_bias2
        self.net['_weight3']=_weight3
        self.net['_bias3']=_bias3
    
    def sigmoid(self, _node):
        return 1/(1+np.exp(-_node))
    
    def forward(self, _node):
        _weight1,_weight2,_weight3=self.net['_weight1'],self.net['_weight2'],self.net['_weight3']
        _bias1,_bias2,_bias3=self.net['_bias1'],self.net['_bias2'],self.net['_bias3']
        _node1=np.dot(_node,_weight1)+_bias1
        _sig1=self.sigmoid(_node1)
        _node2=np.dot(_sig1,_weight2)+_bias2
        _sig2=self.sigmoid(_node2)
        _node3=np.dot(_sig2,_weight3)+_bias3
        _sig3=self.sigmoid(_node3)
        return _sig3


# code
if __name__=='__main__':
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