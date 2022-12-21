import numpy as np

class Base_model:
    def __init__(self):
        pass
    
    def fit(self, data):
        pass
    
    def predict(self, data):
        return np.random.randint(-1, 2, 1)