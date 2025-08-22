"""Anomaly detection model using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetector:
    """Anomaly detector using autoencoder."""
    
    def __init__(self, input_dim: int = 2, threshold: float = 0.95):
        self.model = Autoencoder(input_dim)
        self.threshold = threshold
        self.is_trained = False
    
    def train(self, data: torch.Tensor, epochs: int = 100, batch_size: int = 32):
        """Train the autoencoder."""
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # Forward pass
                outputs = self.model(batch)
                loss = F.mse_loss(outputs, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.is_trained = True
    
    def detect(self, data: torch.Tensor) -> List[float]:
        """Detect anomalies in data."""
        if not self.is_trained:
            # Train on this data if not already trained
            self.train(data)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data)
            # Calculate reconstruction error
            errors = F.mse_loss(reconstructed, data, reduction='none').mean(dim=1)
            # Normalize errors to [0,1]
            normalized_errors = (errors - errors.min()) / (errors.max() - errors.min())
            return normalized_errors.tolist()
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': self.is_trained,
            'threshold': self.threshold
        }, path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        self.threshold = checkpoint['threshold']
