import numpy as np
import torch
def relu(z):
    """
    Args:
        z: (batch_size, hidden_size)
    return:
        a: (batch_size, hidden_size)激活值
    """
    a=np.maximum(z,0)
    
    return a

def derivation_relu(z):
    """
    Args:
        z: (batch_size, hidden_size)
    return:
        dz: (batch_size, hidden_size)导数值
    """
    dz=z.copy()
    dz[dz<=0]=0
    dz[dz>0]=1
    return dz

def sigmoid(z):
    """
    Args:
        z: (batch_size, hidden_size)
    return:
        a: (batch_size, hidden_size)激活值
    """
    a=1/(1+np.exp(-z))
    return a

def bi_cross_entropy(y, y_hat):
    """
    Args:
        y: (batch_size, ) 每个样本的真实label
        y_hat: (batch_size, output_size)， 网络的输出预测得分，已经过sigmoid概率化。output_size即分类类别数
    return:
        loss: scalar
    """
    n_batch = y_hat.shape[0]
    loss = -np.sum(np.log(y_hat)) / n_batch
    return loss
def derivation_sigmoid_cross_entropy(y, y_hat):
    """
    Args:
        logits: (batch_size, output_size)， 网络的输出预测得分, 还没有进行 softmax概率化
        y: (batch_size, ) 每个样本的真实label
    
    Return:
        \frac {\partial C}{\partial z^L}
        (batch_size, output_size)
    """
    y_hat -= 1
    return y_hat

class Network(object):
    """
    fully-connected neural network
    Attributions:
        sizes: list, 输入层、隐藏层、输出层尺寸
        num_layers: 神经网络的层数
        weights: list, 每个元素是一层神经网络的权重
        bias: list, 每个元素是一层神经网络的偏置
        dws: list，存储权重梯度
        dbs: list，存储偏置梯度
        zs: list，存储前向传播临时变量
        _as：list，存储前向传播临时变量
    """
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(i, j) for i, j in zip(self.sizes[:-1], self.sizes[1:])]
        self.bias = [np.random.randn(1, j) for j in self.sizes[1:]]
        self.dws = None
        self.dbs = None
        self.zs = [] 
        self._as = []


    def forward(self, x):
        """
        前向传播
        x: (batch_size, input_size)
        """
        a = x
        self._as.append(a)
        for weight, bias in zip(self.weights[:-1], self.bias[:-1]):
            
            # 计算临时变量z和a并存入self.zs和self._as

            #########################################
            z=np.dot(a,weight)+bias
            self.zs.append(z)
            a=relu(z)
            self._as.append(a)
            #########################################

        logits = np.dot(a, self.weights[-1]) + self.bias[-1]
        y_hat = sigmoid(logits)
        self.zs.append(logits)
        self._as.append(y_hat)
        
        return y_hat

    def backward(self, x, y):
        """
        反向传播
        Args:
            x: (batch_size, input_size)
            y: (batch_size, )
        """

        y_hat = self.forward(x)
        loss = bi_cross_entropy(y, y_hat)

        ################# 反向传播梯度计算 ##############################
        # 输出层误差
        dl = derivation_sigmoid_cross_entropy(y, y_hat)
        # batch的大小
        n = len(x)
        # 最后一层的梯度
        # 每个样本得的梯度求和、求平均
        self.dws[-1] = np.dot(self._as[-2].T, dl) / n
        self.dbs[-1] = np.sum(dl, axis=0, keepdims=True) / n
        for i in range(2, self.num_layers):
            # 计算梯度并存入self.dws和self.dbs，注意矩阵乘法和逐元素乘法
            ############################################################
            dl = np.dot(dl, self.weights[-i+1].T) * derivation_relu(self.zs[-i])  #注意求导形式
            self.dws[-i]=np.dot(self._as[i].T,dl)/n
            self.dbs[-i]=np.sum(dl,axis=0,keepdims=True)/n
            ############################################################
            
        self.zs = [] 
        self._as = []
    
    def zero_grad(self):
        """清空梯度"""
        self.dws = [np.zeros((i, j)) for i, j in zip(self.sizes[:-1], self.sizes[1:])]
        self.dbs = [np.zeros((1, j)) for j in self.sizes[1:]]
        
    def optimize(self, learning_rate):
        """更新梯度"""
        self.weights = [weight - learning_rate * dw for weight, dw in zip(self.weights, self.dws)]
        self.bias = [bias - learning_rate * db for bias, db in zip(self.bias, self.dbs)]

        
def train():
    
    n_batch = 5
    n_input_layer = 2
    n_hidden_layer = 3
    n_output_layer = 1
    n_class = 2
    #x = np.random.rand(n_batch, n_input_layer)
    x=[[3,4],[3,4],[3,4],[3,4],[3,4]]
    #y = np.random.randint(0, n_class, size=n_batch)
    y=[0,1,1,1,0]
    net = Network((n_input_layer, n_hidden_layer, n_output_layer))
    print('initial weights:', net.weights)
    print('initial bias:', net.bias)
    # 执行梯度计算
    
    ##################
    net.zero_grad()
    net.backward(x,y)
    net.optimize(0.5)
    ##################

    print('updated weights:', net.weights)
    print('updated bias:', net.bias)

train()