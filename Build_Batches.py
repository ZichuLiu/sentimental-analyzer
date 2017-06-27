import os
import copy
from scipy.misc import imread
from scipy.misc import imresize
import numpy as np
import pickle
from itertools import chain


class BuildBatches:
    def __init__(self, path, batch_num):
        self.path = path
        self.batch_num = batch_num
        self.dir_list_POS = []
        self.dir_list_NEG = []
        self.dir_list_NEU = []

    def build_mini_batches(self):
        pos = os.listdir(self.path + "/" + "Positive")
        neg = os.listdir(self.path + "/" + "Negative")
        neu = os.listdir(self.path + "/" + "Neutral")

        batch_size = len(neu) // self.batch_num + 1

        for i in range(0, self.batch_num):
            if (i + 1) * batch_size < len(neu):
                self.dir_list_POS.append(pos[i * batch_size:(i + 1) * batch_size])
                self.dir_list_NEG.append(neg[i * batch_size:(i + 1) * batch_size])
                self.dir_list_NEU.append(neu[i * batch_size:(i + 1) * batch_size])
            else:
                self.dir_list_POS.append(pos[i * batch_size:len(pos)])
                self.dir_list_NEG.append(neg[i * batch_size:len(neg)])
                self.dir_list_NEU.append(neu[i * batch_size:len(neu)])

    def dump_files(self, ind):
        os.mkdir(self.path + str(ind))
        dir_list_pos = copy.copy(self.dir_list_POS)
        dir_list_neg = copy.copy(self.dir_list_NEG)
        dir_list_neu = copy.copy(self.dir_list_NEU)

        validation = dict()

        validation["POS"] = []
        validation["NEG"] = []
        validation["NEU"] = []

        for i in range(0, len(dir_list_pos[ind])):
            try:
                validation["POS"].append((imresize(imread(self.path + "Positive" + "/" + dir_list_pos[ind][i]),
                                                   (705, 880, 3)), np.asarray([1, 0, 0])))
            except:
                continue
        for i in range(0, len(dir_list_neg[ind])):
            try:
                validation["NEG"].append((imresize(imread(self.path + "Negative" + "/" + dir_list_neg[ind][i]),
                                                   (705, 880, 3)), np.asarray([0, 0, 1])))
            except:
                continue
        for i in range(0, len(dir_list_neu[ind])):
            try:
                validation["NEU"].append((imresize(imread(self.path + "Neutral" + "/" + dir_list_neu[ind][i]),
                                                   (705, 880, 3)), np.asarray([0, 1, 0])))
            except:
                continue

        dir_list_pos.pop(ind)
        dir_list_neg.pop(ind)
        dir_list_neu.pop(ind)

        print("almost finish" + str(ind))
        dir_list_pos = list(chain(*dir_list_pos))
        dir_list_neg = list(chain(*dir_list_neg))
        dir_list_neu = list(chain(*dir_list_neu))
        print(len(dir_list_pos))
        ind_temp = 0
        while len(dir_list_pos) > 50 and len(dir_list_neu) > 50 and len(dir_list_neg) > 50:

            training = {}
            training["POS"] = []
            training["NEU"] = []
            training["NEG"] = []
            for i in range(0, 50):
                try:
                    training["POS"].append((imresize(imread(self.path + "Positive" + "/" + dir_list_pos.pop(i)),
                                                     (705, 880, 3)), np.asarray([1, 0, 0])))
                except:
                    continue
                try:
                    training["NEG"].append((imresize(imread(self.path + "Negative" + "/" + dir_list_neg.pop(i)),
                                                     (705, 880, 3)), np.asarray([0, 0, 1])))
                except:
                    continue
                try:
                    training["NEU"].append((imresize(imread(self.path + "Neutral" + "/" + dir_list_neu.pop(i)),
                                                     (705, 880, 3)), np.asarray([0, 1, 0])))
                except:
                    continue

            filename_t = self.path + str(ind) + "/" + "Training" + str(ind_temp) + ".pickle"
            print(filename_t)
            with open(filename_t, "wb") as output_file_t:
                pickle.dump(training, output_file_t)
            ind_temp += 1

        filename_v = self.path + str(ind) + "/" + "Validation" + str(ind) + ".pickle"
        with open(filename_v, "wb") as output_file_v:
            pickle.dump(validation, output_file_v)

        print("finished" + str(ind))

    def start(self):
        self.build_mini_batches()
        for i in range(0, self.batch_num):
            self.dump_files(i)
