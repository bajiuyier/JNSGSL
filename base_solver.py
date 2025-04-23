import torch
from copy import deepcopy
import time
from utils import accuracy
from utils import Recorder
from sklearn.metrics import roc_auc_score, r2_score
import torch.nn.functional as F
import numpy as np
import os
import random
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend in non-interactive environments; must be set before importing pyplot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Solver:
    """
    Solver class: responsible for controlling the training, validation, and testing workflow.

    Main functionalities:
      - Initialize data, configuration, and evaluation metrics
      - Choose different training routines depending on single-graph or multi-graph data
      - Manage model saving, evaluation, and early stopping during training
    """

    def __init__(self, conf, dataset):
        # Save config and dataset
        self.dataset = dataset
        self.conf = conf

        # Set device: use GPU by default, unless config specifies CPU
        self.device = torch.device('cuda') if not ('use_cpu' in conf and conf.use_cpu) else torch.device('cpu')

        # Method name (to be set in subclass)
        self.method_name = ''

        # Get number of nodes, features, adjacency matrix, and labels
        self.n_nodes = dataset.n_nodes
        self.feats = dataset.feats
        # Use sparse adjacency matrix directly, or convert to dense if necessary
        self.adj = dataset.adj if self.conf.dataset['sparse'] else dataset.adj.to_dense()
        self.labels = dataset.labels

        # Save auxiliary features (sfeats) if available
        if hasattr(dataset, 'sfeats'):
            self.sfeats = dataset.sfeats

        # Save other dataset properties
        self.dim_feats = dataset.dim_feats
        self.num_targets = dataset.num_targets
        self.n_classes = dataset.n_classes

        # Initialize model and loss function
        self.model = None
        # Use binary cross-entropy loss for binary classification, otherwise use cross-entropy
        self.loss_fn = F.binary_cross_entropy_with_logits if self.num_targets == 1 else F.cross_entropy
        # Use ROC AUC for binary classification, accuracy otherwise
        self.metric = roc_auc_score if self.num_targets == 1 else accuracy
        # For regression tasks (n_classes == 1), use MSELoss and r2_score
        if self.n_classes == 1:
            self.loss_fn = torch.nn.MSELoss()
            self.metric = r2_score

        self.model = None  # Model to be defined in subclass

        # Training, validation, and testing masks (support cross-validation)
        self.train_masks = dataset.train_masks
        self.val_masks = dataset.val_masks
        self.test_masks = dataset.test_masks

        # Current split index
        self.current_split = 0
        self.seed_all(conf.training['seed'])

    def run_exp(self, split=None, debug=False):
        """
        Run experiment: set data split and execute training (for node classification) or graph classification

        Args:
          - split: index of data split, default is 0 if None
          - debug: whether to print debug information

        Returns:
          - training results (dictionary) and other outputs (None)
        """
        # Enable deterministic algorithms if required by config
        if ('use_deterministic' not in self.conf) or self.conf.use_deterministic:
            torch.use_deterministic_algorithms(True)

        # Set current split
        self.set(split)
        # Use learn_nc for node classification, or learn_gc for graph classification (implemented in subclass)
        return self.learn_nc(debug)

    def seed_all(self, seed):
        """
        Set random seeds for reproducibility.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set(self, split):
        """
        Set current data split and initialize training parameters

        Args:
          - split: index of the data split
        """
        if split is None:
            print('split set to default 0.')
            split = 0
        # Check validity of split index
        assert split < len(self.train_masks), 'error, split id is larger than number of splits'

        # Assign masks for train/val/test sets based on split
        self.train_mask = self.train_masks[split]
        self.val_mask = self.val_masks[split]
        self.test_mask = self.test_masks[split]
        self.current_split = split

        # Initialize training helper variables
        self.total_time = 0
        self.best_val_loss = 1e15
        self.weights = None
        self.best_graph = None
        # Initialize result dictionary for train/val/test
        self.result = {'train': -1, 'valid': -1, 'test': -1}
        # Record start time of training
        self.start_time = time.time()
        # Initialize Recorder to monitor validation metric and handle early stopping
        self.recoder = Recorder(self.conf.training['patience'], self.conf.training['criterion'])
        self.adjs = {'ori': self.adj, 'final': self.adj}
        # Call method setup (model and optimizer definition) to be implemented in subclass
        self.set_method()

    def set_method(self):
        """
        Define the model and optimizer.
        This method should be overridden in subclasses.
        """
        self.model = None
        self.optim = None

    def learn_nc(self, debug=False):
        """
        Node classification training routine (single-graph tasks)

        Args:
          - debug: whether to print debug info for each epoch

        Training process:
          1. Iterate through epochs: forward pass, loss calculation, backward pass, and optimization
          2. Evaluate on validation set and record best model using early stopping
          3. Evaluate final model on test set

        Returns:
          - result dictionary with train/val/test metrics, and other outputs (None)
        """
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()  # Set model to training mode
            self.optim.zero_grad()  # Clear gradients

            # Forward pass (input_distributer to be defined in subclass)
            output = self.model(**self.input_distributer())
            # Compute training loss using only training samples
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            # Compute training metric (e.g., accuracy or AUC)
            acc_train = self.metric(self.labels[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())
            # Backward and optimize
            loss_train.backward()
            self.optim.step()

            # Validation phase: compute loss and metric on validation set
            loss_val, acc_val = self.evaluate(self.val_mask)
            # Use Recorder to determine improvement and check early stopping
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            # If validation improves, save current model state
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            # Stop training if early stopping is triggered
            elif flag_earlystop:
                break
            # Print debug info if enabled
            if debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        # Evaluate best model on the test set
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))

        return self.result, None

    def evaluate(self, val_mask):
        """
        Evaluate the model on validation or test set

        Args:
          - val_mask: sample mask for validation or test set

        Returns:
          - loss and metric value
        """
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = self.model(**self.input_distributer())
        # Select only the masked samples for evaluation
        logits = output[val_mask]
        labels = self.labels[val_mask]
        loss = self.loss_fn(logits, labels)
        # Return loss and metric (e.g., accuracy or AUC)
        return loss, self.metric(labels.cpu().numpy(), logits.detach().cpu().numpy())

    def input_distributer(self):
        """
        Input data distributor

        This method prepares and organizes data such as feature matrix and adjacency matrix
        into the format required by the model.
        Placeholder here, should be implemented in specific tasks.
        """
        return None

    def test(self):
        """
        Evaluate the model on the test set:
          1. Load the best saved model weights
          2. Use evaluate() method to compute loss and metrics on test set
        """
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)
