import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear
from utils import convert_to_sparse_coo
from typing import Union
import torch.nn.functional as F


class GNNEncoder_OpenGSL(nn.Module):
    """
    GNNEncoder_OpenGSL: Graph Neural Network Encoder
    This encoder consists of multiple GraphConvolutionLayer layers.
    The final layer outputs n_class dimensions, while intermediate and input layers use n_hidden or n_feat dimensions.
    """
    def __init__(self, n_feat, n_class, n_hidden, n_layers, dropout=0.5,
                 act='F.relu', weight_initializer=None, bias_initializer=None,
                 bias=True, **kwargs):
        """
        Args:
          - n_feat: Dimension of input features
          - n_class: Number of output classes (output dimension of the last layer)
          - n_hidden: Dimension of hidden layers
          - n_layers: Number of GNN layers (default is 2)
          - dropout: Dropout probability to prevent overfitting
          - act: Activation function (string, e.g., 'F.relu')
          - weight_initializer: Method for weight initialization (optional)
          - bias_initializer: Method for bias initialization (optional)
          - bias: Whether to use bias (default is True)
          - **kwargs: Additional optional parameters
        """
        super(GNNEncoder_OpenGSL, self).__init__()
        self.n_feat = n_feat
        self.nclass = n_class
        self.n_layers = n_layers
        self.dropout = dropout
        # Parse activation function (convert string to function using eval)
        self.act = eval(act)
        # Use ModuleList to store the GNN layers
        self.convs = nn.ModuleList()

        # Construct each GNN layer
        for i in range(n_layers):
            # First layer input dimension is n_feat, others are n_hidden
            in_hidden = n_feat if i == 0 else n_hidden
            # Last layer output dimension is n_class, others are n_hidden
            out_hidden = n_class if i == n_layers - 1 else n_hidden
            # Add a GNN layer
            self.convs.append(
                GraphConvolutionLayer(in_hidden, out_hidden, dropout,
                                      act=act, weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer, bias=bias)
            )

    def reset_parameters(self):
        """
        Reset parameters of all GNN layers
        """
        for layer in self.convs:
            layer.reset_parameters()

    def forward(self, x, adj=None):
        """
        Forward propagation

        Args:
          - x: Node feature matrix
          - adj: Adjacency matrix (can be sparse or dense)

        Process:
          1. Pass through each GNN layer in sequence
          2. Apply activation and dropout to all layers except the last

        Returns:
          - Final output of the GNN encoder
        """
        for i, layer in enumerate(self.convs):
            # Feature update through current GNN layer
            x = layer(x, adj)
            # Apply activation and dropout if not the last layer
            if i < self.n_layers - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphConvolutionLayer(nn.Module):
    """
    GraphConvolutionLayer: Single GNN layer
    Applies a linear transformation (MLP) to the input features,
    followed by message passing via the adjacency matrix (neighbor aggregation).
    """
    def __init__(self, in_channels, out_channels, dropout=0.5, bias=True, act='F.relu',
                 weight_initializer=None, bias_initializer=None, **kwargs):
        """
        Args:
          - in_channels: Dimension of input features
          - out_channels: Dimension of output features
          - dropout: Dropout probability
          - bias: Whether to use bias
          - act: Activation function (as a string)
          - weight_initializer: Method for weight initialization (optional)
          - bias_initializer: Method for bias initialization (optional)
        """
        super(GraphConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Linear transformation layer (using PyG's Linear)
        self.mlp = Linear(in_channels, out_channels, bias=bias,
                          weight_initializer=weight_initializer, bias_initializer=bias_initializer)
        self.dropout = dropout
        # Parse activation function from string
        self.act = eval(act)

    def reset_parameters(self):
        """
        Reset parameters of the linear layer
        """
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, adj: Union[SparseTensor, torch.Tensor]):
        """
        Forward pass:
          1. Apply linear transformation to input x
          2. Aggregate features using adjacency matrix (message passing)
          3. Use appropriate matrix multiplication based on the type of adjacency matrix

        Args:
          - x: Node feature matrix
          - adj: Adjacency matrix (can be SparseTensor or torch.Tensor)

        Returns:
          - Updated node feature matrix
        """
        # Apply linear transformation
        x = self.mlp(x)
        # Check type of adjacency matrix
        if isinstance(adj, SparseTensor):
            # Convert to sparse COO format
            adj = convert_to_sparse_coo(adj)
            # Sparse matrix multiplication
            x = torch.sparse.mm(adj, x)
        elif isinstance(adj, torch.Tensor):
            # Dense matrix multiplication
            x = torch.mm(adj, x)
        return x
