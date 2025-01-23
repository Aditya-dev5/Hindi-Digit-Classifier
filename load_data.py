from torchvision.utils import make_grid
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the image
dataset = 'Dataset/Hindi-MNIST'
train_data = dataset + '/' + 'train'  # Replace with your image path
test_data = dataset + '/' + 'test'  # Replace with your image path


# Step 2: Define the transformation
# Convert the image to a PyTorch tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
    transforms.ToTensor(),                        # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize the single channel
])

# Step 2: Load the datasets
train_dataset = datasets.ImageFolder(root=train_data, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data, transform=transform)

# Step 3: Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=17, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=3, shuffle=False, num_workers=4)

# Step 4: Iterate through the DataLoader
for images, labels in train_loader:
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    break

if __name__ == '__main__':
    # View a sample image from training dataset
    tensor_image = train_dataset[4000][0].permute(1, 2, 0)
    image_np = tensor_image.numpy()
    plt.imshow(image_np)
    plt.axis('off')

    #View a batch from training dataset
    np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}')) # to widen the printed array
    for images,labels in train_loader: 
        break #Batch-1

    # First 12 images, labels
    print('Labels: ', labels[:12].numpy())
    im = make_grid(images[:12], nrow=12)  # the default nrow is 8
    plt.figure(figsize=(10,4))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()


