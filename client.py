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

# Definizione del modello CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Definizione delle trasformazioni
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_datasets(partition_id):
    # Caricare il dataset MNIST
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Partizionare il dataset in 5 parti
    partition_sizes = [len(dataset) // 5] * 4 + [len(dataset) - (len(dataset) // 5) * 4]
    datasets = random_split(dataset, partition_sizes)

    # Ottieni il dataset per questo client
    client_dataset = datasets[partition_id]

    # Dividere il dataset del client in train e validation
    train_size = int(0.8 * len(client_dataset))  # 80% per l'addestramento
    val_size = len(client_dataset) - train_size  # 20% per la validazione
    train_dataset, val_dataset = random_split(client_dataset, [train_size, val_size])

    # Creare DataLoader per train e validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader

class FlowerClientWithDP(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)

        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=1.1,  # Controllo del rumore gaussiano
            max_grad_norm=1.0,     # Clipping del gradiente
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(1):  # una epoca
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        # Calcola la media della loss
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
        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * len(images)
        return float(loss) / len(self.val_loader.dataset), len(self.val_loader.dataset), {"loss": float(loss) / len(self.val_loader.dataset)}

def main():
    train_loader, val_loader = load_datasets(args.partition_id)
    model = CNN()
    client = FlowerClientWithDP(model, train_loader, val_loader)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()
