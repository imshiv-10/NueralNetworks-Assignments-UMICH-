# Implement module3.py to test LogicGate class.
# Show your LogicGate class is properly implemented by running do_and, do_nand, do_or, do_nor, and do_xor functions with all possible input combinations.
# Documentation with code using Jupyter Notebook

from logic_gate import LogicGate

Values  = [(0,0),(0,1),(1,0),(1,1)]
sigValues= [0,1]
_logicgateClass = LogicGate()

# Logic gates are elementary building blocks for any digital circuits. 
# It takes one or two inputs and produces output based on those inputs. 
# Outputs may be high (1) or low (0). Logic gates are implemented using diodes or transistors. 
# It can also be constructed using vacuum tubes, electromagnetic elements like optics, molecules, etc. 
# In a computer, most of the electronic circuits are made up of logic gates. Logic gates are used to circuits that perform 
# calculations, data storage, or show off object-oriented programming especially the power of inheritance. 


# 1. AND Gate 
# The AND gate gives an output of 1 if both the two inputs are 1, it gives 0 otherwise.

print("+---------------+----------------+")
print(" | AND Truth Table | Result |")
for i  in Values:
    _result = _logicgateClass.do_and(i[0],i[1])
    print(" A = {}, B = {} | A AND B = {}".format(i[0],i[1],_result)," | ") 


print("+---------------+----------------+")
print(" | NAND Truth Table | Result |")
for i  in Values:
    _result = _logicgateClass.do_nand(i[0],i[1])
    print(" A = {}, B = {} | A NAND B = {}".format(i[0],i[1],_result)," | ") 


print("+---------------+----------------+")
print(" | OR Truth Table | Result |")
for i  in Values:
    _result = _logicgateClass.do_or(i[0],i[1])
    print(" A = {}, B = {} | A OR B = {}".format(i[0],i[1],_result)," | ")    


print("+---------------+----------------+")
print(" | XOR Truth Table | Result |")
for i  in Values:
    _result = _logicgateClass.do_xor(i[0],i[1])
    print(" A = {}, B = {} | A XOR B = {}".format(i[0],i[1],_result)," | ")    


print("+---------------+----------------+")
print(" | NOT Truth Table | Result |")
for i  in range(len(sigValues)):
    _result = _logicgateClass.do_not(sigValues[i])
    print(" A = {} |        |       A NOT = {}".format(sigValues[i],_result))    


print("+---------------+----------------+")
print(" | XNOR Truth Table | Result |")
for i  in Values:
    _result = _logicgateClass.do_xnor(i[0],i[1])
    print(" A = {}, B = {} | A XNOR B = {}".format(i[0],i[1],_result)," | ") 
