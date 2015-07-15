
# coding: utf-8

# In[219]:

import os, csv


# In[220]:

def arrayParser(arr):
    # CSV cast string list to python list
    smooth_stage_1 = arr.replace('[', '').replace(']', '').split(',')
    smooth_stage_2 = map(lambda unit: float(unit), smooth_stage_1)
    return smooth_stage_2


# In[221]:

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


# In[222]:

class LVQNet:
    def __init__(self, inCount, outCount):
        self.inputs   = inCount
        self.outputs  = outCount
        self.alpha    = 0.1
        self.csvCount = 0  # Limit the csvs input to number of output neurons
        self.neurons  = {} # Numbered index map (to outputs)
              
        for n in range(outCount):
            curr_neuron = LVQNeuron(n)
            self.neurons[n] = curr_neuron
        
    def enterCSV(self, filepath):
        if self.csvCount >= self.outputs:
            print "Reached limit of neurons"
            return
            
        with open(filepath, 'r') as f:
            read = csv.reader(f, delimiter=',')
            row = read.next()
            curr_neuron = self.neurons[self.csvCount]
            curr_neuron.setWeights(row[1])
            self.csvCount += 1
            
            print "Successfully added neuron from CSV", filepath
            return
    
    def __len__(self):
        return len(self.neurons)
    
    def edist(self, inputs, weights):
        # Euclidean Distance helper function 
        euclideanDistance = 0
        
        if len(inputs) != len(weights):
            return
        
        for i in inputs:
            nth = inputs[i] - weights[i]
            nth = nth ** 2
            euclideanDistance += nth
            
        return euclideanDistance ** (0.5)
    
    def minDist(self, inputVector):
        for neuron in self.neurons:
            


# In[223]:

class LVQData:
    def __init__(self):
        self.data = [] # list of tuples 
    
    def loadCSV(self, filepath, label):
        # Will skip first line of each CSV since LVQ initializes using the first lines
        with open(filepath, 'r') as f:
            read = csv.reader(f, delimiter=',')
            read.next()
            
            for row in read:
                # Tuple with STFT bins and then the label
                data_struct = (arrayParser(row[1]), label)
                self.data.append(data_struct)
                
        return self.data


# In[224]:

if __name__ == '__main__':
    # Create Network with in and out neuron parameters
    koho = LVQNet(1025, 3)
    
    # Enter data (1-1 CSV to Output Neurons) Initializes the neurons with first onset
    koho.enterCSV('./data/snareFrames.csv')
    koho.enterCSV('./data/kickDrumFrames.csv')
    
    # Instantiate LVQ Training Data Structure and load rest of CSVs with labels
    dataset = LVQData()
    dataset.loadCSV('./data/snareFrames.csv', 'snare')
    dataset.loadCSV('./data/kickDrumFrames.csv', 'kickdrum')
    

