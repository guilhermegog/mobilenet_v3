import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from mobilenet_v3 import mobilenet_v3_large

# Define the transformations for the train and test sets
transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Load the CIFAR-10 datasets
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=12
)
# Instantiate the network, loss function, and optimizer
device = torch.device("cuda:2")
model = mobilenet_v3_large()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(
                f"[Epoch {epoch + 1}, Batch {i +
                                             1}] loss: {running_loss / 100:.3f}"
            )
            running_loss = 0.0


print("Finished Training")
h_net = torch.nn.Sequential(*list(model.features.children())[:8])

torch.save(h_net.state_dict(), 'h_net.pth')
print(h_net)
