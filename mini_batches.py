import os
import pickle


class MiniBatches:

    def __init__(self,path):
        self.index = 0
        self.path = path
        self.training_batches = [batch for batch in os.listdir(path) if "Validation" not in batch]
        self.validation_batch = [batch for batch in os.listdir(path) if "Validation" in batch][0]

    def get_train(self):
        temp = self.index
        training_sample = []
        inputdata = pickle.load(open(self.path + self.training_batches[temp],"rb"))
        training_sample.append([tup[0] for tup in (inputdata["POS"]+inputdata["NEG"]+inputdata["NEU"]) if
                                tup[0].shape == (705,880,3)])
        training_sample.append([tup[1] for tup in (inputdata["POS"]+inputdata["NEG"]+inputdata["NEU"]) if
                                tup[0].shape == (705,880,3)])
        if self.index == len(self.training_batches)-1:
            self.index = 0
        else:
            self.index += 1
        return training_sample

    def get_validation(self):
        validation_sample = []
        inputdata = pickle.load(open(self.path + self.validation_batch,"rb"))
        validation_sample.append([tup[0] for tup in (inputdata["POS"] + inputdata["NEG"] + inputdata["NEU"]) if
                                tup[0].shape == (705, 880, 3)])
        validation_sample.append([tup[1] for tup in (inputdata["POS"] + inputdata["NEG"] + inputdata["NEU"]) if
                                tup[0].shape == (705, 880, 3)])
        return validation_sample
