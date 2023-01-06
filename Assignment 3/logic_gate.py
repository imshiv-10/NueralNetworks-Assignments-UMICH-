# do_and, do_nand, do_or, do_nor, do_xor, do_nxor and do_not.
# Logic gate member functions must be implemented using Numpy. 
# Perceptron
 
# LogicGate's Implementation through class definitions
import numpy as np

class LogicGate:
    def __init__(self):
        pass

# AND Gate 
# The AND gate gives an output of 1 if both the two inputs are 1, it gives 0 otherwise.     
    def do_and(self, x1,x2):
        # w1, w2 are weights of the paths to reach the destination to y 
        # th ---> b threshold value 
        #w1, w2 , b  = 0.5, 0.5 , 0.8
        _nodes = np.array([x1,x2])
        _weights = np.array([1,1])
        _bias = -1
        # if y = 0, x1.w1 + x2.w2 < b
        # if y = 1, x1.w1 + x2.w2 > b
        #_eval = x1*w1 + x2*w2
        _eval = np.sum([_nodes*_weights]) + _bias
        return 1 if _eval>0 else 0     
    
# NAND Gate 
# The NAND gate (negated AND) gives an output of 0 if both inputs are 1, it gives 1 otherwise. 
    def do_nand(self, x1,x2):
       return 0 if self.do_and(x1, x2) else 1

#OR Gate 
#The OR gate gives an output of 1 if either of the two inputs is 1, it gives 0 otherwise
    def do_or(self,x1,x2):
        # w1, w2 are weights of the paths to reach the destination to y 
        # th ---> b threshold value 
        #w1, w2 , b  = 0.5, 0.5 , 0.8
        _nodes = np.array([x1,x2])
        _weights = np.array([1,1])
        _bias = -1
        # if y = 0, x1.w1 + x2.w2 < b
        # if y = 1, x1.w1 + x2.w2 > b
        #_eval = x1*w1 + x2*w2
        _eval = np.sum([_nodes*_weights]) + _bias
        return 1 if _eval>0 else 0     

# NOR Gate 
# The XOR gate gives an output of 1 if either of the inputs is different, it gives 0 if they are the same. 
    def do_xor(self,x1,x2):
        # w1, w2 are weights of the paths to reach the destination to y 
        # th ---> b threshold value 
        #w1, w2 , b  = 0.5, 0.5 , 0.2
        # if y = ~(0), x1.w1 + x2.w2 < b
        # if y = ~(1), x1.w1 + x2.w2 > b
        y1 = self.do_or(x1, x2)               
        y2 = self.do_nand(x1,x2)
        y =   self.do_and(y1, y2)
        return y               

# NOT Gate 
# It acts as an inverter. It takes only one input. If the input is given as 1, it will invert the result as 0 and vice-versa. 
    def do_not(self, x):
        # w are weights of the paths to reach the destination to y 
        # th ---> b threshold value 
        #w, b  = 1 , 0.8
        # if y = 0, x.w < b
        # if y = 1, x.w > b
        #_eval = x*w
        _nodes = np.array(x)
        _weights = np.array(1)
        _bias = -1
        _eval = np.array(_nodes* _weights) + _bias
        return 1 if _eval else 0

# XOR Gate 
# The NOR gate (negated OR) gives an output of 1 if both inputs are 0, it gives 0 otherwise.
    def do_nor(self,x1,x2):
        return 1 if self.do_or(x1, x2)<0 else 0                   

#XNOR Gate 
# The XNOR gate (negated XOR) gives an output of 1 if both in the puts are the same and 0 if both are different. 
    def do_xnor(self,x1,x2):
        # w1, w2 are weights of the paths to reach the destination to y 
        # threshold-> b thresold value 
       return 0 if self.do_xor(x1, x2) else 1 


# Driver code
if __name__=='__main__':
    _nand = LogicGate.do_not(0)
    print(_nand)