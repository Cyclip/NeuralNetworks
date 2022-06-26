from defaultFuncs import randomName

class Neuron:
    """A singular neuron.
    Contains a value and a bias, and should be in a layer.
    It can optionally be given a name, but all non-named neurons
     will have randomly generated names.
    """
    def __init__(self, bias, value=0, name=None):
        """Construct a singular neuron.

        Args:
            bias (float): Bias to add onto the weighted sum of output
            value (int, optional): Value of the neuron. Defaults to 0.
            name (string, optional): Name of the neuron. Defaults to None.
        """
        self.value = value
        self.bias = bias
        
        # connections *to* this neuron
        self.connections = []
        
        if name:
            self.name = name
        else:
            self.name = "Neuron-" + randomName(n=4)
    
    def get_name(self):
        return self.name
    
    def activate(self, normalizationFunction):
        """Set the value of the neuron based off of its connections

        Args:
            normalizationFunction (function): Function to normalize with
        """
        # calculate weighted sum
        self.value = normalizationFunction(sum([
            connection.neuronFrom.value * connection.weight
            for connection in self.connections
        ]) + self.bias)

    def __str__(self):
        return f"{self.get_name()}: {self.value} (bias {self.bias})"


class IONeuron(Neuron):
    """Either input or output neurons
    """
    def __init__(self, name):
        super().__init__(0, name=name)


class Connection:
    """The connection from one neuron in layer n to another neuron in layer (n+1)
    Contains a weight and a target layer (for debugging)
    """
    def __init__(self, neuronFrom, weight, fromLayer):
        self.neuronFrom = neuronFrom
        self.weight = weight
        self.fromLayer = fromLayer + 1
        self.toLayer = fromLayer + 2
    
    def __str__(self):
        return f"[{self.weight:<5}] {self.neuronTo.get_name()}[{self.toLayer}]"