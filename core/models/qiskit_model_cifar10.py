import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
import torch.nn.functional as F


class QISKITMODEL(nn.Module):
    """Baseline network for MNIST dataset.

    This network is the implement of baseline network for MNIST dataset, from paper
    `BadNets: Evaluating Backdooring Attackson Deep Neural Networks <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8685687&tag=1>`_.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = nn.Linear(84,3)  # 1-dimensional output from QNN
        self.qnn = TorchConnector(self.create_qnn())  # Apply torch connector, weights chosen
        self.fc4 = nn.Linear(4,4)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc4(x)
        return x

    def create_qnn(self):
        feature_map = ZZFeatureMap(3)
        ansatz = RealAmplitudes(3, reps=1)
        qc = QuantumCircuit(3)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
        parity = lambda x: "{:b}".format(x).count("1") % 4  # optional interpret function
        output_shape = 4  # parity = 0, 1
        qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
            interpret=parity,
            output_shape=output_shape,
        )
        return qnn


# if __name__ == '__main__':
#     baseline_MNIST_network = BaselineMNISTNetwork()
#     x = torch.randn(16, 1, 28, 28)
#     x = baseline_MNIST_network(x)
#     print(x.size())
#     print(x)
