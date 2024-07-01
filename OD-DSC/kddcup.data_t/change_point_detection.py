import numpy as np
from sklearn.metrics import mean_squared_error
import Partial_training_online
import Change_point_train


class Change_point(object):
    def __init__(self,data, R_threshold, path_str, parameters):
        self.data = data
        self.R_threshold = R_threshold
        self.path_str = path_str
        self.parameters = parameters
        self.R_Pre = 0
        self.R_latter = 0

    def get_RMSE(self, y_int, y_out):
        R = np.sqrt(((y_int - y_out) ** 2).mean())
        return R


    def judgment(self):
        global R_Pre
        global Train_index
        dim = self.data.shape[1]
        if int(self.path_str) == 1:      #Data that is currently the first timestamp
            print("==========================="+ self.path_str+ "=============================")
            Train_index = str(self.parameters['C_point_index'])
            Train = Change_point_train.Train(self.data)
            Train.get_Cpoint_train_coef(self.data, Train_index)
            #Partial training
            Y, L = Partial_training_online.get_coef_NChange_test(self.data, self.path_str, C_point_index=Train_index,
                                                             parameter=self.parameters)
            R_Pre = self.get_RMSE(self.data, np.array(Y))
            print("29", R_Pre)

        elif int(self.path_str) > 1:
            print("===========================" + self.path_str + "=============================")
            print("C_point_index========", Train_index, self.path_str)
            Y, L = Partial_training_online.get_coef_NChange_test(self.data, self.path_str, C_point_index=Train_index,
                                                                 parameter=self.parameters)
            self.R_latter = self.get_RMSE(self.data, np.array(Y))
            print("35", R_Pre, self.R_latter)
            self.R_Diff = abs(self.R_latter - R_Pre)
            print("________________", self.path_str, self.R_Diff)
            if self.R_Diff >self.R_threshold:
                print("change point_index", self.path_str)
                Train_index = str(self.parameters['C_point_index'])
                #Train = Change_point_train.Train(self.data)
                #Train.get_Cpoint_train_coef(self.data, Train_index)
                Y, L = Partial_training_online.get_coef_NChange_test(self.data, self.path_str,
                                                                     C_point_index=Train_index,
                                                                     parameter=self.parameters)
                self.R_latter = self.get_RMSE(self.data, np.array(Y))
                print("current_R", self.R_latter)
                R_Pre = self.R_latter
            else:
                R_Pre = self.R_latter
        return Y, L








