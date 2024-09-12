
import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Scalar, FitRes, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F


# Definizione del modello
class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Aggiungi un testloader globale sul server
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class FedAvgServer(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        """Aggregate model weights using weighted average."""
        if not results:
            return None

        # Convert results
        weights_results = [
            (self.parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Aggregate weights
        aggregated_weights = self.aggregate_weights(weights_results)
         # Convert aggregated weights to a state_dict
        model = OptimizedCNN()
        state_dict = self.list_to_state_dict(aggregated_weights, model)
        
        # Carica il modello con lo state_dict aggregato
        model.load_state_dict(state_dict)

        # Valuta il modello aggregato sui dati globali
        self.evaluate_aggregated_model(model, rnd)

        # Convert weights back to parameters
        parameters_aggregated = self.weights_to_parameters(aggregated_weights)

        return parameters_aggregated, {}
    
    def list_to_state_dict(self, weights: List[np.ndarray], model: nn.Module):
        """Convert a list of numpy arrays to a PyTorch state_dict."""
        state_dict = model.state_dict()  # Ottieni un template dello state_dict del modello
        state_dict_keys = list(state_dict.keys())  # Prendi le chiavi dello state_dict

        # Associa ogni chiave dello state_dict ai pesi corrispondenti
        new_state_dict = {key: torch.tensor(weights[i]) for i, key in enumerate(state_dict_keys)}

        return new_state_dict
    
    def evaluate_aggregated_model(self, model, rnd):
        """Valuta il modello aggregato e genera una confusion matrix."""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcola e visualizza la confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix - Round {rnd}')
        plt.savefig(f'confusion_matrix_round_{rnd}.png')
        plt.close()

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and accuracies using weighted average."""
        if not results:
            return None, {}

        # Collect losses and accuracies
        losses = []
        accuracies = []
        examples = []
        for _, evaluate_res in results:
            losses.append(evaluate_res.loss)
            accuracies.append(evaluate_res.metrics["accuracy"])
            examples.append(evaluate_res.num_examples)

        # Calculate weighted averages
        avg_loss = np.average(losses, weights=examples)
        avg_accuracy = np.average(accuracies, weights=examples)

        # Print the results
        print(f"Round {rnd} - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

        return avg_loss, {"accuracy": avg_accuracy}

    def weights_to_parameters(self, weights: List[np.ndarray]) -> Parameters:
        return fl.common.ndarrays_to_parameters(weights)

    def parameters_to_weights(self, parameters: Parameters) -> List[np.ndarray]:
        return fl.common.parameters_to_ndarrays(parameters)

    def aggregate_weights(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: List[np.ndarray] = [
            np.sum(layer_updates, axis=0) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime

# Create strategy
strategy = FedAvgServer(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
)

# Start server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
