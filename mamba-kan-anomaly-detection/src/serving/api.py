"""FastAPI inference server for Mamba-KAN anomaly detection.

Serves the trained model as a REST API for real-time sensor anomaly scoring.
Supports both single-window and batch inference with configurable thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from src.models.mamba_kan import MambaKANDetector


# ── Request / Response Schemas ───────────────────────────────

class SensorWindow(BaseModel):
    """Single sensor reading window."""
    values: list[list[float]] = Field(
        ..., description="2D array of shape (window_size, n_features)"
    )

class PredictionRequest(BaseModel):
    """Batch of sensor windows for anomaly scoring."""
    windows: list[SensorWindow]
    threshold: float | None = Field(None, description="Override anomaly threshold")

class AnomalyResult(BaseModel):
    """Anomaly detection result for a single window."""
    score: float
    is_anomaly: bool
    timestep_scores: list[float]

class PredictionResponse(BaseModel):
    """Batch prediction response."""
    results: list[AnomalyResult]
    model_name: str
    threshold_used: float


# ── Application Factory ──────────────────────────────────────

class ModelServer:
    """Encapsulates model loading and inference logic."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model: MambaKANDetector | None = None
        self.threshold: float = 0.5
        self._checkpoint_path = checkpoint_path

    def load(self, input_dim: int = 8, hidden_dim: int = 64) -> None:
        """Load model from checkpoint."""
        self.model = MambaKANDetector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        path = Path(self._checkpoint_path)
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"No checkpoint at {path}, using random weights")

        self.model.eval()

    @torch.no_grad()
    def predict(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a batch of windows.

        Args:
            windows: Array of shape (batch, window_size, n_features).

        Returns:
            Tuple of (window_scores, timestep_scores).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        tensor = torch.from_numpy(windows).float().to(self.device)
        _, scores = self.model(tensor)

        timestep_scores = scores.cpu().numpy()
        window_scores = timestep_scores.mean(axis=-1)

        return window_scores, timestep_scores


def create_app(
    checkpoint_path: str = "checkpoints/best_model.pt",
    default_threshold: float = 0.5,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Mamba-KAN Anomaly Detector",
        description="Real-time sensor anomaly detection using hybrid Mamba-KAN architecture",
        version="0.1.0",
    )

    server = ModelServer(checkpoint_path)

    @app.on_event("startup")
    async def startup() -> None:
        server.load()
        server.threshold = default_threshold

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy", "model_loaded": str(server.model is not None)}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        try:
            batch = np.array([w.values for w in request.windows], dtype=np.float32)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")

        threshold = request.threshold or server.threshold
        window_scores, timestep_scores = server.predict(batch)

        results = [
            AnomalyResult(
                score=float(window_scores[i]),
                is_anomaly=bool(window_scores[i] >= threshold),
                timestep_scores=timestep_scores[i].tolist(),
            )
            for i in range(len(window_scores))
        ]

        return PredictionResponse(
            results=results,
            model_name="mamba_kan",
            threshold_used=threshold,
        )

    @app.get("/model/info")
    async def model_info() -> dict[str, Any]:
        if server.model is None:
            return {"error": "Model not loaded"}
        return {
            "parameters": server.model.count_parameters(),
            "threshold": server.threshold,
        }

    return app


def main() -> None:
    """Launch the inference server."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
