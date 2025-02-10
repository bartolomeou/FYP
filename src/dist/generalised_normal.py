import numpy as np


class GeneralNormal():
    def __init__(self, mu=0, alpha=1, beta=2):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def logpi(self, x):
        return - (np.abs(x - self.mu) / self.alpha)**self.beta

    def first_logpi(self, x):
        diff = x - self.mu
        return - (self.beta / self.alpha**self.beta) * np.abs(diff)**(self.beta-1) * np.sign(diff)

    def second_logpi(self, x):
        diff = x - self.mu
        return - ((self.beta * (self.beta-1)) / self.alpha**self.beta) * np.abs(diff)**(self.beta-2)

    def third_logpi(self, x):
        diff = x - self.mu
        return - ((self.beta * (self.beta-1) * (self.beta-2)) / self.alpha**self.beta) * np.abs(diff)**(self.beta-3) * np.sign(diff)
    

class SmoothedGeneralNormal():
    def __init__(self, mu=0, alpha=1, beta=2, epsilon=1):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def logpi(self, x):
        return - (np.sqrt(self.epsilon* + (x - self.mu)**2) / self.alpha)**self.beta

    def first_logpi(self, x):
        diff = x - self.mu
        return - (self.beta / self.alpha**self.beta) * diff * (self.epsilon + diff**2)**((self.beta-2)/2)

    def second_logpi(self, x):
        diff = x - self.mu
        return - (self.beta / self.alpha**self.beta) * (self.epsilon + diff**2)**((self.beta-4)/2) * ((self.beta-1) * diff**2 + self.epsilon)

    def third_logpi(self, x):
        diff = x - self.mu
        return - ((self.beta * (self.beta-2)) / self.alpha**self.beta) * diff * (self.epsilon + diff**2)**((self.beta-6)/2) * ((self.beta-1) * diff**2 + 3*self.epsilon)
    
    def sample(self, size):
        pass
        