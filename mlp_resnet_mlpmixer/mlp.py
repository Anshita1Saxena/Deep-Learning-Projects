import torch
from typing import List, Tuple
from torch import nn


class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
       
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Cuda Environment
        # From slide 48: 02_NN_training.pdf
        b = torch.sqrt(torch.ones(1, device="cuda") * 6) / torch.sqrt(
            torch.ones(1, device="cuda") * in_features + torch.ones(1, device="cuda") * out_features)
        self.weight = torch.nn.Parameter(
            ((-2 * b) * torch.rand(out_features, in_features, device="cuda") + b))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, device="cuda"))
        # Without CUDA for TA tests
        # b = torch.sqrt(torch.ones(1) * 6) / torch.sqrt(
        #     torch.ones(1) * in_features + torch.ones(1) * out_features)
        # self.weight = torch.nn.Parameter(
        #     ((-2 * b) * torch.rand(out_features, in_features) + b))
        # self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        """
            :param input: [bsz, in_features]
            :return result [bsz, out_features]
        """

        return torch.matmul(input, self.weight.t()) + self.bias


class MLP(torch.nn.Module):
    # 20 points
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)
        
        # Initializaton
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        # Instantiation matters, if we keep [] then the accuracy will be low because it doesn't sense the correct type
        hidden_layers = nn.ModuleList()
        hidden_layers.append(Linear(input_size, hidden_sizes[0]))
        for s in range(len(hidden_sizes)-1):
            hidden_layer = Linear(hidden_sizes[s], hidden_sizes[s+1])
            hidden_layers.append(hidden_layer)
        output_layer = Linear(hidden_sizes[len(hidden_sizes)-1], num_classes)
        return (hidden_layers, output_layer)
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        # nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model. nn.functional.relu on the
        # other side is just the functional API call to the relu function, so that you can add it e.g. in your forward
        # method yourself.
        if activation == 'tanh':
            return torch.tanh(inputs)
        elif activation == 'relu':
            return torch.relu(inputs)
        elif activation == 'sigmoid':
            return torch.sigmoid(inputs)

    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """
        # You are registering your parameter properly, but you should use nn.Module.named_parameters rather than
        # nn.Module.parameters to access the names.
        # for k, v in module.named_parameters():
        #     if k == 'weight':
        #         nn.init.xavier_normal_(v)
        #     if k == 'bias':
        #         nn.init.constant_(v.data, 0)
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors.
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        images = images.view(images.size(0), -1)
        for s in self.hidden_layers:
            images = self.activation_fn(self.activation, s(images))
        return self.output_layer(images)
