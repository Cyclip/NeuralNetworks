import sys
sys.path.insert(0, "../src")
from main import Layer, Network, Neuron, Connection, IONeuron

inputNeurons = [
    IONeuron("inp1"),
    IONeuron("inp2"),
    IONeuron("inp3"),
]

# input layer
nn = Network(inputNeurons)

# hidden layer
nn.add_layer(
    Layer((
        # biases
        Neuron(1),
        Neuron(0),
        Neuron(-1),
        Neuron(1),
    ))
)

# output layer
nn.add_layer(
    Layer((
        IONeuron("out1"),
        IONeuron("out2"),
    ))
)

nn.set_inputs((0, 1, -1))
nn.build_connections()

nn.start()

for output in nn.get_outputs():
    print(output)