
# coding: utf-8

# In[91]:

import random


# In[116]:

class Node:
    def __init__(self, name):
        self.name = name
        self.weight = random.uniform(-1, 1)
        
    def __str__(self):
        return self.name + " -- " + str(self.weight)


# In[117]:

class LVQNet:
    def __init__(self):
        self.alpha           = 0.10
        self.iterationsLeft  = 100
        self.outputLayerSize = 0
        self.outputLayer     = {}
    
    def addNode(self, name):
        curr_node = Node(name)
        self.outputLayer[name] = curr_node
        self.outputLayerSize += 1
    
    def __str__(self):
        return "Output Layer consists of " + str(self.outputLayerSize) + " nodes."


# In[118]:

class InputVector:
    idGen = 1
    
    def __init__(self, infoTuple):
        self.idKey    = InputVector.idGen
        self.vector   = infoTuple[0]
        self.category = infoTuple[1]
        
        InputVector.idGen += 1
        
    def __str__(self):
        return str(self.id) + " : " + str(self.info)


# In[119]:

class TrainingSet:
    # A Dictionary of IDs that map to input vectors
    #   => Tuples of Input Vectors ([Information], "Classifier")
    def __init__(self):
        self.inputs = {}
    
    # Input must be InputVector type
    def addInput(self, vector):
        v = InputVector(vector)
        self.inputs[v.idKey] = vector
        
    def __str__(self):
        return str(self.inputs)


# In[120]:

t_set = TrainingSet()

# Examples of Input Vectors
t_set.addInput(([1, 2, 3, 4], "Kick Drum"))
t_set.addInput(([2, 3, 4, 5], "Cymbals"))


# In[121]:

single_drum = ["kick", "high-tom", "low-tom", "snare", "hi-hat"] # grab bag of some drums we sampled

outcome_nodes = [] # create single and combos of drums

for drum in single_drum:
    outcome_nodes.append(drum)
    
for drum in range(len(single_drum)):
    for other_drum in range(drum, len(single_drum)):
        if not(drum == other_drum):
            outcome_nodes.append(single_drum[drum] + " and " + single_drum[other_drum])
            
print outcome_nodes, len(outcome_nodes)

