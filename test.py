import torch
from torch import nn
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model.mnist.cnn import CNN_mnist, AggregateModel

# MNIST Dataset transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=Subset(train_dataset,range(1000)), batch_size=64, shuffle=True)

num_local_models = 3
local_models = [CNN_mnist(1, 10) for _ in range(num_local_models)]

aggregate_model = AggregateModel(local_models, output_channels=10)

def train_model(model, data_loader, num_epochs):
    global loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training Complete")


# Train the model
train_model(aggregate_model, train_loader, num_epochs=1)
# train_model(mnist_cnn(1, 10), train_loader, num_epochs=5)
print("Aggregation Weights for Conv1:", aggregate_model.get_aggregation_weights())


