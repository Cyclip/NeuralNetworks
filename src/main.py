import random
from neuron import Neuron, Connection, IONeuron
from defaultFuncs import randomStrength, sigmoid, doProbability

class Layer:
    def __init__(self, neurons):
        self.nextLayer = None
        self.neurons = neurons
    
    def set_next_layer(self, nextLayer):
        self.nextLayer = nextLayer


class Network:
    """Foundations for a neural network

    1. Create a neural network (with given input neurons)
    2. Add additional middle layers
    3. Add a final output layer
    4. Set inputs
    5. Run
    """
    def __init__(self, inputNeurons):
        """Construct a neural network from a given list of input

        Args:
            inputNeurons (list[IONeuron]): List of input neurons
        """
        self.layers = [Layer(inputNeurons),]
    
    def add_layer(self, layer):
        """Add a layer to the neural network

        Args:
            layer (Layer): Layer to add
        """
        self.layers.append(layer)
    
    def build_connections(self, strengthFunction=randomStrength):
        """Construct the connections for the neural network
        This should be completed once all layers are added.

        Args:
            strengthFunction (function, optional): Function to generate weights for connections. Defaults to randomStrength.
        """
        # for each layer (skipping input layer)
        for layerIndex, layer in enumerate(self.layers[1:]):
            # accomodate for skipping first layer
            layerIndex += 1

            # for each output neuron in current layer
            for outputNeuron in layer.neurons:
                # print(f"\nLayer {layerIndex} {outputNeuron.name}")
                # iterate through all input neurons (previous layer)
                for inputNeuron in self.layers[layerIndex - 1].neurons:
                    # create a new connection with a base strength
                    connection = Connection(inputNeuron, strengthFunction(), layerIndex)
                    outputNeuron.connections.append(connection)
    
    def set_inputs(self, values):
        """Set the inputs for the neural network

        Args:
            values (list[float/int]): List of inputs
        """
        for inputNeuron, value in zip(self.layers[0].neurons, values):
            inputNeuron.value = value
    
    def set_input(self, name, value):
        """Set the input of a single input neuron.

        Args:
            name (string): Name of input neuron
            value (float): Value to set it to
        """
        for inputNeuron in self.layers[0].neurons:
            if inputNeuron.name == name:
                inputNeuron.value = value
                return
    
    def start(self, normalizationFunction=sigmoid):
        """Run the neural network.
        For each layer it will iterate through every neuron.
        For every neuron, it will have given connections from the previous layer.
        The output is derived from calculating the weighted sum of the previous neurons * their respective weight
        + the current neuron's bias, all of which is passed through a given normalization function.

        Args:
            normalizationFunction (function, optional): Function to normalize outputs. Defaults to sigmoid.
        """
        # for each layer (skipping input layer)
        for layerIndex, layer in enumerate(self.layers[1:]):
            # accomodate for skipping first layer
            layerIndex += 1

            # for each output neuron in the current layer
            for outputNeuron in layer.neurons:
                # calculate an output passed through a normalizing function
                outputNeuron.activate(normalizationFunction)
    
    def get_outputs(self):
        """Receive an iterator of all output neurons

        Returns:
            iterator: Iterator containing output neurons
        """
        return iter(self.layers[-1].neurons)
    
    def get_highest_output(self):
        highest = [None, float('-inf')]

        for outputNeuron in self.get_outputs():
            if outputNeuron.value > highest[1]:
                highest = [outputNeuron.name, outputNeuron.value]
        
        return highest
    
    def mutate(self, chance, standardDeviation):
        """Mutate each neuron biases and weights

        Args:
            chance (float): Mutation rate (0.0 to 1.0)
            standardDeviation (float): Standard deviation for gaussian number
        """
        # for each layer (skipping input layer)
        for layer in self.layers[1:]:
            # for each neuron, have a chance to modify:
            # - bias
            # - each connection

            for neuron in layer.neurons:
                if doProbability(chance):
                    neuron.bias += random.gauss(0, standardDeviation)
            
                for connection in neuron.connections:
                    if doProbability(chance):
                        connection.weight += random.gauss(0, standardDeviation)
    
    def __str__(self):
        s = ""
        for layerIndex, layer in enumerate(self.layers):
            s += f"Layer {layerIndex}\n"

            for neuron in layer.neurons:
                s += f"\tNeuron {neuron.name:<12}: {neuron.value}\n"
                s += "\t\t" + "\t".join(
                    [str(round(i.weight, 3)) for i in neuron.connections]
                ) + "\n\n"
        
        s += "\n\n"
        return s
