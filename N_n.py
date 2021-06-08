import numpy as np

class N_n():

    def __init__(self, input_layer, hidden_layer, output_layer):
        """Конструктор"""
        self.alpha = 0.4
        self.w_input = 2 * np.random.random((hidden_layer, input_layer)) - 1
        self.w_hidden = 2 * np.random.random((output_layer, hidden_layer)) - 1
        self.T_input = np.zeros((1, hidden_layer))
        self.T_hidden = np.zeros((1, output_layer))
        self.value_hidden = np.zeros((1, hidden_layer))
        self.value_output = np.zeros((1, output_layer))
        self.gamma_hidden = np.zeros((1, hidden_layer))
        self.gamma_output = np.zeros((1, output_layer))
        self.E = np.zeros(output_layer)

    def change_w_hidden(self):
        """Зміна вейтів на скритому шарі"""
        self.w_hidden = self.w_hidden - np.dot(np.transpose(self.alpha * self.value_output 
            * (1 - self.value_output) * self.gamma_output), self.value_hidden)

    def change_w_input(self, vector):
        """Зміна вейтів на вхідному шарі"""
        self.w_input = self.w_input - np.dot(np.transpose(self.alpha * self.value_hidden 
            * (1 - self.value_hidden) * self.gamma_hidden), vector)
    
    def change_T_hidden(self):
        """Зміна погорових значень на скритому шарі"""
        self.T_hidden = self.T_hidden + self.alpha * self.value_output * (1 - self.value_output) * self.gamma_output

    def change_T_input(self):
        """Зміна погорових значень на вхідному шарі"""
        self.T_input = self.T_input + self.alpha * self.value_hidden * (1 - self.value_hidden) * self.gamma_hidden
        
        
