"""Updated tests for anomaly detection using 1D Kp-like inputs."""

import pytest
import torch
import numpy as np
from app.ml.anomaly_detector import AnomalyDetector, Autoencoder


def test_autoencoder_architecture():
    model = Autoencoder(input_dim=1)
    x = torch.randn(10, 1)
    output = model(x)
    assert output.shape == x.shape


def test_anomaly_detector_training():
    detector = AnomalyDetector(input_dim=1)
    normal_data = torch.randn(200, 1) * 0.05  # Tight cluster
    detector.train(normal_data, epochs=15)
    assert detector.is_trained


def test_anomaly_detection_kp_values():
    detector = AnomalyDetector(input_dim=1, threshold=0.9)
    normal_data = torch.randn(300, 1) * 0.05 + 3.0  # Simulate typical Kp around 3
    detector.train(normal_data, epochs=20)

    # Normal segment
    segment = normal_data[:10]
    normal_scores = detector.detect(segment)
    assert all(score <= detector.threshold for score in normal_scores)

    # Inject anomaly (very high Kp = 15)
    anomalous = torch.tensor([[15.0]])
    anomaly_scores = detector.detect(anomalous)
    assert any(score > detector.threshold for score in anomaly_scores)


def test_model_save_load(tmp_path):
    detector = AnomalyDetector(input_dim=1)
    normal_data = torch.randn(150, 1) * 0.05 + 2.5
    detector.train(normal_data, epochs=10)
    path = tmp_path / "kp_model.pt"
    detector.save_model(str(path))
    assert path.exists()

    new_detector = AnomalyDetector(input_dim=1)
    new_detector.load_model(str(path))
    assert new_detector.is_trained

    test_batch = torch.randn(5, 1) * 0.05 + 2.5
    orig_scores = detector.detect(test_batch)
    loaded_scores = new_detector.detect(test_batch)
    assert np.allclose(orig_scores, loaded_scores)
