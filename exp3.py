import numpy as np
import scipy.misc as sc

class Exp3:

    def __init__(self, num_of_arms, context):
        self.gamma = min(1.0, np.sqrt(np.log(num_of_arms) / num_of_arms))
        self.weights = np.zeros(num_of_arms)
        self.num_of_arms = num_of_arms
        self.last_arm = 0
        self.t = 1
        self.context = context


    def predict(self):

        if self.gamma != 1.0:
            c = sc.logsumexp(-self.gamma * self.weights)
            prob_dist = np.exp((-self.gamma * self.weights) - c)
        else:
            prob_dist = np.ones(self.num_of_arms) / np.float(self.num_of_arms)
        arm = np.random.choice(a=self.num_of_arms, p=prob_dist)
        self.last_arm = arm
        self.arm_counter[arm] += 1
        return arm


    def update(self, reward, last_arm = None):

        if last_arm == None:
            last_arm = self.last_arm
        loss = 1 - reward
        if self.gamma != 1.0:
            c = sc.logsumexp(-self.gamma * self.weights)
            prob = np.exp((-self.gamma * self.weights[last_arm]) - c)
        else:
            prob = 1.0 / np.float(self.num_of_arms)

        assert prob > 0

        estimated = loss / prob
        self.weights[last_arm] += estimated
        self.t += 1
        self.gamma = min(1.0, np.sqrt(np.log(self.num_of_arms) / (self.num_of_arms * self.t)))
