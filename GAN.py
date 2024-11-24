import numpy as np
import matplotlib.pyplot as plt

class GAN:
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=1):
        # Initialize weights for Generator
        self.G_W1 = np.random.randn(input_dim, hidden_dim)
        self.G_b1 = np.zeros(hidden_dim)
        self.G_W2 = np.random.randn(hidden_dim, output_dim)
        self.G_b2 = np.zeros(output_dim)

        # Initialize weights for Discriminator
        self.D_W1 = np.random.randn(output_dim, hidden_dim)
        self.D_b1 = np.zeros(hidden_dim)
        self.D_W2 = np.random.randn(hidden_dim, 1)
        self.D_b2 = np.zeros(1)

    def generator(self, z):
        # Hidden layer
        G_h1 = np.tanh(z @ self.G_W1 + self.G_b1)
        # Output layer
        G_out = G_h1 @ self.G_W2 + self.G_b2
        return G_out

    def discriminator(self, x):
        # Hidden layer
        D_h1 = np.tanh(x @ self.D_W1 + self.D_b1)
        # Output layer
        D_out = 1 / (1 + np.exp(-(D_h1 @ self.D_W2 + self.D_b2)))
        return D_out

    def generator_gradient(self, z, g_sample, d_prob):
        # Gradients for Generator
        d_G_W2 = np.zeros_like(self.G_W2)
        d_G_b2 = np.zeros_like(self.G_b2)
        d_G_W1 = np.zeros_like(self.G_W1)
        d_G_b1 = np.zeros_like(self.G_b1)
        
        # Calculate gradients
        G_h1 = np.tanh(z @ self.G_W1 + self.G_b1)
        
        # Output layer gradients
        d_G_W2 = G_h1.T @ (-(1-d_prob))
        d_G_b2 = np.sum(-(1-d_prob), axis=0)
        
        # Hidden layer gradients
        d_G_h1 = (-(1-d_prob)) @ self.G_W2.T
        d_G_W1 = z.T @ (d_G_h1 * (1 - G_h1**2))
        d_G_b1 = np.sum(d_G_h1 * (1 - G_h1**2), axis=0)
        
        return d_G_W1, d_G_b1, d_G_W2, d_G_b2

    def discriminator_gradient(self, x_real, x_fake, d_prob_real, d_prob_fake):
        # Gradients for Discriminator
        d_D_W2 = np.zeros_like(self.D_W2)
        d_D_b2 = np.zeros_like(self.D_b2)
        d_D_W1 = np.zeros_like(self.D_W1)
        d_D_b1 = np.zeros_like(self.D_b1)
        
        # Real data
        D_h1_real = np.tanh(x_real @ self.D_W1 + self.D_b1)
        # Fake data
        D_h1_fake = np.tanh(x_fake @ self.D_W1 + self.D_b1)
        
        # Output layer gradients
        d_D_W2 = (D_h1_real.T @ (-(1-d_prob_real)) + D_h1_fake.T @ (-d_prob_fake))
        d_D_b2 = np.sum(-(1-d_prob_real), axis=0) + np.sum(-d_prob_fake, axis=0)
        
        # Hidden layer gradients
        d_D_h1_real = (-(1-d_prob_real)) @ self.D_W2.T
        d_D_h1_fake = (-d_prob_fake) @ self.D_W2.T
        
        d_D_W1 = (x_real.T @ (d_D_h1_real * (1 - D_h1_real**2)) + 
                  x_fake.T @ (d_D_h1_fake * (1 - D_h1_fake**2)))
        d_D_b1 = (np.sum(d_D_h1_real * (1 - D_h1_real**2), axis=0) + 
                  np.sum(d_D_h1_fake * (1 - D_h1_fake**2), axis=0))
        
        return d_D_W1, d_D_b1, d_D_W2, d_D_b2

    def train(self, epochs=10000, batch_size=128, learning_rate=0.0002):
        losses = []
        
        for epoch in range(epochs):
            # Generate random noise
            z = np.random.randn(batch_size, 100)
            # Generate fake data
            g_sample = self.generator(z)
            
            # Generate real data (simple Gaussian distribution)
            x_real = np.random.normal(4, 1.5, size=(batch_size, 1))
            
            # Discriminator probabilities
            d_prob_real = self.discriminator(x_real)
            d_prob_fake = self.discriminator(g_sample)
            
            # Calculate gradients
            d_G_W1, d_G_b1, d_G_W2, d_G_b2 = self.generator_gradient(z, g_sample, d_prob_fake)
            d_D_W1, d_D_b1, d_D_W2, d_D_b2 = self.discriminator_gradient(x_real, g_sample, 
                                                                         d_prob_real, d_prob_fake)
            
            # Update Generator parameters
            self.G_W1 -= learning_rate * d_G_W1
            self.G_b1 -= learning_rate * d_G_b1
            self.G_W2 -= learning_rate * d_G_W2
            self.G_b2 -= learning_rate * d_G_b2
            
            # Update Discriminator parameters
            self.D_W1 -= learning_rate * d_D_W1
            self.D_b1 -= learning_rate * d_D_b1
            self.D_W2 -= learning_rate * d_D_W2
            self.D_b2 -= learning_rate * d_D_b2
            
            if epoch % 1000 == 0:
                loss = np.mean(np.log(d_prob_real) + np.log(1 - d_prob_fake))
                losses.append(loss)
                print(f'Epoch: {epoch}, Loss: {loss}')
        
        return losses

    def generate_samples(self, num_samples=1000):
        z = np.random.randn(num_samples, 100)
        return self.generator(z)

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