import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

np.random.seed(0)
def plot_loss(iter_i, Loss):
    plt.figure("Loss vs Iter")
    plt.plot(iter_i ,Loss,'b*')
    plt.pause(0.1)

class Optimizations:
    """
    Class containing different optimization functions
    Currenty available: sgd-> Stocastic Gradient Descent
    """
    def __init__(self):
        pass
    
    @staticmethod
    def SGD(data, output, learning_rate = 0.1, max_iter = 100, converge = 10**-6):
        """ 
        SGD using sum of square loss
        """
        #initializing the weights and bias
        #add bias coefficent as 1 to data
        data = np.concatenate([data, np.ones((data.shape[0],1))], axis = 1)
        weights = np.random.rand(data.shape[1]).reshape(1, data.shape[1])
        
        # update weights and bias using sgd
        prv_loss = np.inf
        for i in range(max_iter):
            h_theta = np.sum(data*weights, axis=1)
            Loss =  (h_theta - output).reshape(1,-1)
            A = (learning_rate/Loss.shape[1]) * (Loss @ data)
            weights = weights - A
            Loss = 0.5*np.mean(Loss**2)  
            print(Loss)
            if(np.abs(prv_loss - Loss) < converge):
                print("Function converged")
                break
            else:
                prv_loss = Loss
            if(i == max_iter-1):
                print("Maximum iteration reached")

        return weights

class linear_regression:
    def __init__(self):
        self.data = None
        self.output = None
        self.weights = None
        self.alpha = 0.01
        self.iter = 100
        self.bias = None
        
        
    def run(self):
        assert(self.data is not None)
        assert(self.data.shape[0] > 2 and self.data.shape[1] > 0)

        #Update weights using stocastic gradient disecent
        self.weights = Optimizations.SGD(self.data, self.output, learning_rate=self.alpha, max_iter=self.iter)
    
    def predict(self, input_data):
        assert(self.weights is not None)
        assert(input_data.shape[1] == self.weights.shape[1]-1)
        output_data = (self.weights[:, 0:-1]@input_data.T) + self.weights[:, -1]
        return output_data

if __name__ == "__main__":
    lr = linear_regression()
    #read the csv file
    housing = pd.read_csv("../dataset/housing.csv")
    housing_numpy  = housing.loc[:,['RM', 'LSTAT', 'PTRATIO','MEDV']].values[0:400, :]
    
    #normalizing the values such between 0 and 1
    MEAN = np.mean(housing_numpy, axis=0)
    STD = np.std(housing_numpy, axis=0)
    housing_norm = (housing_numpy - MEAN)/(STD)

    #applying linear regression
    lr.data = housing_norm[:,0:-1]
    lr.output = housing_norm[:,-1]
    lr.run()

    #Perform prediction
    housing_test  = housing.loc[:,['RM', 'LSTAT', 'PTRATIO','MEDV']].values[400:450, :]
    housing_test = (housing_test - MEAN)/(STD)
    predicted_output = lr.predict(housing_test[:,:-1])
    predicted_output = (predicted_output*STD)+MEAN




  



    