
# coding: utf-8

# In[10]:

import os, csv


# In[11]:

class LVQNeuron:
    def __init__(self, name):
        self.name = name
        self.weights = []
    
    def setWeights(self, weights):
        weights = weights.replace('[', '').replace(']', '').split(',')
        
        for weight in weights:
            self.weights.append(float(weight))

    def __len__(self):
        return len(self.weights)


# In[12]:

class LVQNet:
    def __init__(self, inCount, outCount):
        self.inputs   = inCount
        self.outputs  = outCount
        self.alpha    = 0.1
        self.csvCount = 0  # Limit the CSVs input to number of output neurons
        self.neurons  = {} # Numbered index map (to outputs)
              
        for n in range(outCount):
            curr_neuron = LVQNeuron(n)
            self.neurons[n] = curr_neuron
            
    def __len__(self):
        return len(self.neurons)
    
    def getWeights(self, neuronNo):
        return self.neurons[neuronNo].weights
    
    def setWeights(self, neuronNo, weights):
        oldWeights = self.neurons[neuronNo]
        if len(oldWeights) != len(weights):
            return
        else:
            self.neurons[neuronNo] = weights
            return
        
    # STEP 0
    def enterCSV(self, filepath): 
        if self.csvCount >= self.outputs:
            print "Reached limit of neurons" # (TODO) Throw error
            return
            
        with open(filepath, 'r') as f:
            read = csv.reader(f, delimiter=',')
            row = read.next()
            curr_neuron = self.neurons[self.csvCount]
            curr_neuron.setWeights(row[1])
            self.csvCount += 1
            
            print "Successfully added neuron from CSV", filepath
            return

    # STEP 3.1
    def edist(self, inputs, weights):
        euclideanDistance = 0
        
        if len(inputs) != len(weights):
            print len(inputs), "different length than", len(weights) # (TODO) Error
            return
        
        for i in range(len(inputs)):
            nth = inputs[i] - weights[i]
            nth = nth ** 2
            euclideanDistance += nth
             
        return euclideanDistance ** (0.5)
    
    # STEP 3.2 
    def minDist(self, inputVector):
        scores = [] # Euclidean Distances
        
        for neuron in self.neurons:
            wunit = self.neurons[neuron].weights
            scores.append(self.edist(inputVector, wunit))
        
        minNeuronIndex = scores.index(min(scores))
        return minNeuronIndex
    
    # STEP 4 
    def calibrate(self, neuron, guessNo, inputVector):
        guess   = dataset.lookupInstrument(guessNo)
        weights = self.getWeights(guessNo) 
        
        addfunc = lambda oldWeight, vec: oldWeight + self.alpha * (vec - oldWeight)
        subfunc = lambda oldWeight, vec: oldWeight - self.alpha * (vec - oldWeight)

        if neuron == guess: 
            # print "Right Guess"
            newWeights = map(addfunc, weights, inputVector) # assign weights as new weights
            self.setWeights(guessNo, newWeights)
            return newWeights
        else:
            # print "Wrong Guess"
            newWeights = map(subfunc, weights, inputVector) # assign weights as new weights
            self.setWeights(guessNo, newWeights)
            return newWeights


# In[13]:

class LVQData:
    def __init__(self):
        self.data          = [] # list of tuples 
        self.instrumentMap = {} # map integers with instruments (labels)
        self.instrumentNum = 0  # current integer instrument (to neuron)
    
    def loadCSV(self, filepath, label):
        with open(filepath, 'r') as f:
            read = csv.reader(f, delimiter=',')
            read.next()
            
            for row in read:
                # Tuple with STFT bins and then the label
                data_struct = (arrayParser(row[1]), label)
                self.data.append(data_struct)
        
            self.instrumentMap[self.instrumentNum] = label
            self.instrumentNum += 1
            
        return self.data
    
    def getVector(self, index):
        return self.data[index][0]
    
    def getVectorLabel(self, index):
        return self.data[index][1]
    
    def lookupInstrument(self, index):
        return self.instrumentMap[index]


# In[14]:

def arrayParser(arr):
    # CSV usage: cast string list to python list
    smooth_stage_1 = arr.replace('[', '').replace(']', '').split(',')
    smooth_stage_2 = map(lambda unit: float(unit), smooth_stage_1)
    return smooth_stage_2


# In[15]:

### Driver: Outline of the API / Algorithm in use    
if __name__ == '__main__':
    # Create Network with in and out neuron parameters
    koho = LVQNet(1025, 2)
    
    # Enter data (1-1 CSV to Output Neurons) Initializes the neurons with first onset
    koho.enterCSV('./data/snareFrames.csv')
    koho.enterCSV('./data/kickDrumFrames.csv')
    
    # Instantiate LVQ Training Data Structure and load rest of CSVs with labels
    dataset = LVQData()
    dataset.loadCSV('./data/snareFrames.csv', 'snare')
    dataset.loadCSV('./data/kickDrumFrames.csv', 'kick-drum')
    
    # (TODO) Put the koho minDists in loop / logic
    guess = koho.minDist(dataset.getVector(2))  # using a specific sample frame (snare)
    guess2 = koho.minDist(dataset.getVector(8)) # using another sample frame (kick)
 
    newWeights = koho.calibrate(dataset.getVectorLabel(8), guess2, dataset.getVector(8))
    print newWeights


# In[ ]:



