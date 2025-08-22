"""Optimized anomaly detection model using PyTorch and NumPy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
from sklearn.neural_network import MLPRegressor


class Autoencoder(nn.Module):
    """Lightweight autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class OptimizedAnomalyDetector:
    """Fast anomaly detector using scikit-learn MLPRegressor."""
    
    def __init__(self, threshold_percentile: float = 95):
        self.model = MLPRegressor(
            hidden_layer_sizes=(8,), 
            max_iter=200, 
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.is_trained = False
    
    def extract_tle_features(self, tle_triplets: List[dict]) -> np.ndarray:
        """Extract optimized numeric features from TLE data."""
        features = []
        for tle in tle_triplets:
            try:
                name, line1, line2 = tle["name"], tle["line1"], tle["line2"]
                
                # Extract meaningful TLE features
                feature_vector = [
                    len(name),
                    len(line1), 
                    len(line2),
                    int(line1[18:20]) if len(line1) > 20 else 0,  # Epoch year
                    float(line1[20:32]) if len(line1) > 32 else 0.0,  # Epoch day
                    float(line2[26:33]) if len(line2) > 33 else 0.0,  # Inclination
                    float(line2[34:42]) if len(line2) > 42 else 0.0,  # RAAN
                ]
                features.append(feature_vector)
            except (ValueError, IndexError):
                # Fallback for malformed TLE
                features.append([len(tle.get("name", "")), 
                               len(tle.get("line1", "")), 
                               len(tle.get("line2", "")), 0, 0, 0, 0])
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize features to prevent any single feature from dominating
        max_vals = features.max(axis=0)
        max_vals[max_vals == 0] = 1  # Prevent division by zero
        return features / max_vals
    
    def train(self, tle_data: List[dict], subsample_ratio: float = 0.1):
        """Train on a subsample of TLE data for speed."""
        features = self.extract_tle_features(tle_data)
        
        # Subsample for faster training
        n_samples = max(10, int(len(features) * subsample_ratio))
        indices = np.random.choice(len(features), n_samples, replace=False)
        train_features = features[indices]
        
        # Train autoencoder (input = output for unsupervised learning)
        self.model.fit(train_features, train_features)
        
        # Calculate threshold based on reconstruction errors
        reconstructed = self.model.predict(train_features)
        errors = np.mean((train_features - reconstructed) ** 2, axis=1)
        self.threshold = np.percentile(errors, self.threshold_percentile)
        self.is_trained = True
    
    def detect(self, tle_data: List[dict]) -> List[float]:
        """Detect anomalies with optimized feature extraction."""
        if not self.is_trained:
            self.train(tle_data)
        
        features = self.extract_tle_features(tle_data)
        reconstructed = self.model.predict(features)
        errors = np.mean((features - reconstructed) ** 2, axis=1)
        
        # Normalize errors to [0,1] for consistent thresholding
        if errors.max() > errors.min():
            normalized_errors = (errors - errors.min()) / (errors.max() - errors.min())
        else:
            normalized_errors = np.zeros_like(errors)
            
        return normalized_errors.tolist()


class AnomalyDetector:
    """Legacy PyTorch-based detector for compatibility."""
    
    def __init__(self, input_dim: int = 2, threshold: float = 0.95):
        self.model = Autoencoder(input_dim)
        self.threshold = threshold
        self.is_trained = False
    
    def train(self, data: torch.Tensor, epochs: int = 50, batch_size: int = 32):
        """Train the autoencoder with reduced epochs for speed."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
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
            self.train(data)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data)
            errors = F.mse_loss(reconstructed, data, reduction='none').mean(dim=1)
            if errors.max() > errors.min():
                normalized_errors = (errors - errors.min()) / (errors.max() - errors.min())
            else:
                normalized_errors = torch.zeros_like(errors)
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
