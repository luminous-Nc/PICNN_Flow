import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model.new_model import PICNN
import matplotlib.pyplot as plt
from LBMDataset import LBMDataset



if __name__ == "__main__":
    # Prepare your dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = LBMDataset(root_dir='dataset', transform=transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PICNN().to(device)

    # Define the split indices for training and validation set
    train_indices = list(range(1000))
    valid_indices = list(range(1000, len(dataset)))

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    # Define data loaders
    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=16, sampler=valid_sampler)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # Training
        model.train()
        for inputs, targets_x, targets_y in train_loader:
            inputs, targets_x, targets_y = inputs.to(device), targets_x.to(device), targets_y.to(device)  # Move data to GPU

            optimizer.zero_grad()
            output1, output2 = model(inputs)

            loss = criterion(output1, targets_x) + criterion(output2, targets_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_indices)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            for inputs, targets_x, targets_y in valid_loader:
                inputs, targets_x, targets_y = inputs.to(device), targets_x.to(device), targets_y.to(
                    device)  # Move data to GPU
                output1, output2 = model(inputs)
                loss = criterion(output1, targets_x) + criterion(output2, targets_y)
                valid_loss += loss.item() * inputs.size(0)

        valid_loss /= len(valid_indices)
        valid_losses.append(valid_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    # Plot the loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
