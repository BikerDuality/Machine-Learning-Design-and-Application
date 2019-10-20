import numpy as np
class singlePerception:
    def __init__(self, inputs,targets):
        self.nDate=len(inputs)
        self.weights=np.random.rand(inputs.shape[1]+1,1)    

    # 训练感知机
    def train(self, inputs, targets, eta, nIterations):
        # 偏置加入输入，构成4 * 3维矩阵
        inputs = np.concatenate((inputs, - np.ones((self.nDate, 1))), axis=1)
        # 训练
        for n in range(nIterations):
            # 前向传播
            self.outputs = self.forward(inputs)
            # 修改权值
            self.weights = self.weights + eta * \
                np.dot(np.transpose(inputs), targets - self.outputs)

    # 前向传播
    def forward(self, inputs):
        outputs = np.dot(inputs, self.weights)
        return np.where(outputs > 0.5, 1, 0)  # 输出阈值

    # 输出
    def prn(self):
        print("Percetron's weights:\n", self.weights)
        print("Percetron's outputs:\n", self.outputs)

inputs=np.array([[0,0],[0,1],[1,0],[1,1]])
targets=np.array([[0],[0],[0],[1]])
p=singlePerception(inputs,targets)
p.train(inputs,targets,0.25,6)
p.prn()