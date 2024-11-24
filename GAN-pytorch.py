import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=1):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
        self.criterion = nn.BCELoss()

    def train(self, epochs=10000, batch_size=128):
        losses = []

        for epoch in range(epochs):
            # Generate random noise
            z = torch.randn(batch_size, 100).to(self.device)
            
            # Generate fake data
            fake_data = self.generator(z)
            
            # Generate real data (simple Gaussian distribution)
            real_data = torch.normal(4, 1.5, size=(batch_size, 1)).to(self.device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            d_output_real = self.discriminator(real_data)
            d_loss_real = self.criterion(d_output_real, real_labels)
            
            d_output_fake = self.discriminator(fake_data.detach())
            d_loss_fake = self.criterion(d_output_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()

            # Train Generator
            self.g_optimizer.zero_grad()
            
            g_output = self.discriminator(fake_data)
            g_loss = self.criterion(g_output, real_labels)
            
            g_loss.backward()
            self.g_optimizer.step()

            if epoch % 1000 == 0:
                loss = d_loss.item() + g_loss.item()
                losses.append(loss)
                print(f'Epoch: {epoch}, Loss: {loss}')

        return losses

    def generate_samples(self, num_samples=1000):
        with torch.no_grad():
            z = torch.randn(num_samples, 100).to(self.device)
            samples = self.generator(z)
        return samples.cpu().numpy()

# Training
if __name__ == "__main__":
    gan = GAN()
    losses = gan.train()

    # Generate samples
    samples = gan.generate_samples()

    # Plot results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch (x1000)')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=50)
    plt.title('Generated Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()