import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import flwr as fl
from opacus import PrivacyEngine

# Parsing degli argomenti da riga di comando
parser = argparse.ArgumentParser(description='Federated Learning Client')
parser.add_argument('--partition-id', type=int, required=True, help='Partition ID')
args = parser.parse_args()

# Definire le trasformazioni
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Caricare il dataset MNIST
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Partizionare il dataset in 5 parti
partition_sizes = [len(dataset) // 5] * 4 + [len(dataset) - (len(dataset) // 5) * 4]
datasets = random_split(dataset, partition_sizes)

# Creare DataLoader per il client specifico
loader = DataLoader(datasets[args.partition_id], batch_size=32, shuffle=True)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FlowerClientWithDP(fl.client.NumPyClient):
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0.0  # Inizializza la loss totale
        for _ in range(1):  # Una epoca
            for images, labels in self.loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()  # Accumula la loss
                loss.backward()
                self.optimizer.step()
        average_loss = total_loss / len(self.loader)  # Calcola la loss media
        return self.get_parameters(config), len(self.loader.dataset), {"loss": average_loss}  # Restituisci la loss media

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in self.loader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        return float(loss) / len(self.loader), len(self.loader.dataset), {"accuracy": float(correct) / len(self.loader.dataset)}
# Inizializzare il modello e il client
model = SimpleNN()
client = FlowerClientWithDP(model, loader)

# Avviare il client di Flower
fl.client.start_client(server_address="localhost:8080", client=client.to_client())
