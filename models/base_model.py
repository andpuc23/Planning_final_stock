import numpy as np

class Base_model:
    def __init__(self):
        pass
    
    def fit(self, data):
        print('called base_model.fit(data), did nothing')
        pass
    
    def fit(self, state, action, reward):
        print('called base_model.fit(state, action, reward), did nothing')
        pass
    
    def predict(self, data):
        return np.random.randint(-1, 2, 1)
    