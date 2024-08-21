import argparse
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import flwr as fl
from opacus import PrivacyEngine
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated")

parser = argparse.ArgumentParser(description='Federated Learning Client')
parser.add_argument('--partition-id', type=int, required=True, help='Partition ID')
args = parser.parse_args()

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_datasets(partition_id, num_partitions=30):
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)
    
     # Partiziona il dataset in 30 parti
    partition_size = len(mnist_train) // num_partitions
    partitions = random_split(mnist_train, [partition_size] * num_partitions)
    
    # Seleziona la partizione per questo client
    partition = partitions[partition_id]
    
    # Dividi in train (80%) e validation (20%)
    train_size = int(0.8 * len(partition))
    val_size = len(partition) - train_size
    train_subset, val_subset = random_split(partition, [train_size, val_size])
    
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=32)
    testloader = DataLoader(mnist_test, batch_size=32)
    
    return trainloader, valloader, testloader

class FlowerClientWithDP(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, delta=1e-5):
        self.model = model
        self.train_loader = trainloader
        self.val_loader = valloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Inizializzazione del PrivacyEngine con delta
        
        self.privacy_engine = PrivacyEngine()

        # Collegamento del PrivacyEngine al modello e all'ottimizzatore
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=1.1,  # Controllo del rumore
            max_grad_norm=1.0,  # Clipping del gradiente
        )

        # Delta per la privacy differenziale
        self.delta = delta

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(5):
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        # Calcola l'epsilon dopo l'addestramento
        epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
        print(f"Training complete. (ε = {epsilon:.2f}, δ = {self.delta})")

        loss_sum = 0
        with torch.no_grad():
            for images, labels in self.train_loader:
                outputs = self.model(images)
                loss_sum += self.criterion(outputs, labels).item() * len(images)
        average_loss = loss_sum / len(self.train_loader.dataset)
        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": average_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * len(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = correct / total
        avg_loss = loss / len(self.val_loader.dataset)
        return avg_loss, len(self.val_loader.dataset), {"loss": avg_loss, "accuracy": accuracy}

def main():
    trainloader, valloader, testloader = load_datasets(args.partition_id)
    model = OptimizedCNN()
    client = FlowerClientWithDP(model, trainloader, valloader)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()
