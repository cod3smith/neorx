"""
Surrogate docking model — fast approximation of DockBot binding affinity.

Problem:
    AutoDock Vina takes 5-30 seconds per docking call.  An RL training
    run with 10,000+ docking evaluations would take ~28 hours.

Solution:
    Train an MLP surrogate that predicts binding affinity from:
    - Morgan fingerprint of the ligand (2048-bit → float)
    - Protein-target embedding (from R-GCN node embedding, 128-D)

    The surrogate runs in <1ms per call, enabling real-time RL.

Workflow:
    1. **Seed phase**: Run real DockBot on N initial (molecule, target)
       pairs to build a training set.
    2. **Train surrogate**: Fit the MLP on ``(fingerprint, target_emb) → affinity``.
    3. **RL phase**: Agent uses surrogate for reward during training.
    4. **Recalibration**: Periodically run real DockBot on agent's best
       molecules and retrain the surrogate (prevents distribution shift).

Reference:
    Gentile et al. (2020) "Deep Docking: A Deep Learning Platform for
    Augmentation of Structure Based Drug Discovery", ACS Central Science.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────── #
#  Molecular Fingerprint Utilities                                           #
# ────────────────────────────────────────────────────────────────────────── #


def smiles_to_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> NDArray[np.floating] | None:
    """Convert SMILES to Morgan fingerprint vector.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    radius : int
        Morgan fingerprint radius (default 2 ≈ ECFP4).
    n_bits : int
        Fingerprint length.

    Returns
    -------
    ndarray of shape ``(n_bits,)`` or ``None`` if parsing fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────── #
#  Surrogate Docking Model                                                   #
# ────────────────────────────────────────────────────────────────────────── #


class _DockingMLP(nn.Module):
    """MLP that predicts binding affinity from fingerprint + target embedding."""

    def __init__(
        self,
        fp_dim: int = 2048,
        target_dim: int = 128,
        hidden_dim: int = 256,
        n_hidden: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_dim = fp_dim + target_dim

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_hidden - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SurrogateDockingModel:
    """Fast docking score predictor for RL reward computation.

    Predicts binding affinity (kcal/mol) from molecular fingerprints
    and target protein embeddings.  Trained from real DockBot results.

    Parameters
    ----------
    fp_dim : int
        Fingerprint dimension (default 2048 for ECFP4).
    target_dim : int
        Target embedding dimension (default 128 from R-GCN).
    hidden_dim : int
        MLP hidden layer size.
    lr : float
        Learning rate for training.
    device : str | None
        ``"cuda"`` or ``"cpu"``.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        fp_dim: int = 2048,
        target_dim: int = 128,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.fp_dim = fp_dim
        self.target_dim = target_dim
        self.lr = lr

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = _DockingMLP(
            fp_dim=fp_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Training data buffer for recalibration
        self._buffer_fps: list[NDArray[np.floating]] = []
        self._buffer_targets: list[NDArray[np.floating]] = []
        self._buffer_affinities: list[float] = []
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Whether the surrogate has been trained with real data."""
        return self._is_trained

    # ------------------------------------------------------------------ #
    #  Data Collection                                                     #
    # ------------------------------------------------------------------ #

    def add_observation(
        self,
        fingerprint: NDArray[np.floating],
        target_embedding: NDArray[np.floating],
        real_affinity: float,
    ) -> None:
        """Add a real docking observation to the training buffer.

        Parameters
        ----------
        fingerprint : ndarray ``(fp_dim,)``
            Morgan fingerprint of the ligand.
        target_embedding : ndarray ``(target_dim,)``
            R-GCN embedding of the target node.
        real_affinity : float
            Actual binding affinity from DockBot (kcal/mol).
        """
        self._buffer_fps.append(fingerprint.copy())
        self._buffer_targets.append(target_embedding.copy())
        self._buffer_affinities.append(real_affinity)

    def add_smiles_observation(
        self,
        smiles: str,
        target_embedding: NDArray[np.floating],
        real_affinity: float,
    ) -> bool:
        """Convenience: add an observation from SMILES string.

        Returns ``True`` if the SMILES was valid and added.
        """
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            return False
        self.add_observation(fp, target_embedding, real_affinity)
        return True

    @property
    def buffer_size(self) -> int:
        """Number of observations in the training buffer."""
        return len(self._buffer_affinities)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        epochs: int = 200,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> float:
        """Train the surrogate on buffered observations.

        Parameters
        ----------
        epochs : int
            Training epochs.
        batch_size : int
            Mini-batch size.
        verbose : bool
            Print loss every 50 epochs.

        Returns
        -------
        float
            Final training MSE loss.
        """
        if self.buffer_size < 10:
            logger.warning(
                "Surrogate has only %d observations — need at least 10 to train.",
                self.buffer_size,
            )
            return float("inf")

        fps = np.array(self._buffer_fps, dtype=np.float32)
        tgts = np.array(self._buffer_targets, dtype=np.float32)
        affs = np.array(self._buffer_affinities, dtype=np.float32)

        # Concatenate fingerprint + target embedding
        X = np.concatenate([fps, tgts], axis=1)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        Yt = torch.tensor(affs, dtype=torch.float32, device=self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        n = len(Xt)
        final_loss = float("inf")

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                x_batch = Xt[idx]
                y_batch = Yt[idx]

                pred = self.model(x_batch)
                loss = loss_fn(pred, y_batch)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            final_loss = epoch_loss / max(n_batches, 1)

            if verbose and epoch % 50 == 0:
                logger.info("Surrogate epoch %d/%d — MSE: %.4f", epoch, epochs, final_loss)

        self.model.eval()
        self._is_trained = True
        logger.info(
            "Surrogate trained on %d observations — final MSE: %.4f",
            self.buffer_size, final_loss,
        )
        return final_loss

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(
        self,
        fingerprint: NDArray[np.floating],
        target_embedding: NDArray[np.floating],
    ) -> float:
        """Predict binding affinity for a single (ligand, target) pair.

        Parameters
        ----------
        fingerprint : ndarray ``(fp_dim,)``
        target_embedding : ndarray ``(target_dim,)``

        Returns
        -------
        float
            Predicted binding affinity (kcal/mol, negative = better).
        """
        if not self._is_trained:
            # Return neutral estimate before training
            return -5.0

        x = np.concatenate([fingerprint, target_embedding])
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.model(xt).item())

    @torch.no_grad()
    def predict_smiles(
        self,
        smiles: str,
        target_embedding: NDArray[np.floating],
    ) -> float | None:
        """Convenience: predict affinity from SMILES string.

        Returns ``None`` if SMILES parsing fails.
        """
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            return None
        return self.predict(fp, target_embedding)

    @torch.no_grad()
    def predict_batch(
        self,
        fingerprints: NDArray[np.floating],
        target_embeddings: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Predict affinities for a batch.

        Parameters
        ----------
        fingerprints : ndarray ``(B, fp_dim)``
        target_embeddings : ndarray ``(B, target_dim)``

        Returns
        -------
        ndarray ``(B,)``
        """
        if not self._is_trained:
            return np.full(len(fingerprints), -5.0, dtype=np.float32)

        X = np.concatenate([fingerprints, target_embeddings], axis=1)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.model(Xt).cpu().numpy()

    # ------------------------------------------------------------------ #
    #  Recalibration                                                       #
    # ------------------------------------------------------------------ #

    def recalibrate(
        self,
        fingerprints: NDArray[np.floating],
        target_embeddings: NDArray[np.floating],
        real_affinities: NDArray[np.floating],
        epochs: int = 100,
    ) -> float:
        """Recalibrate with new real docking data.

        Adds the new data to the buffer and retrains.  Prevents
        distribution shift when the RL agent explores regions
        far from the initial training data.

        Parameters
        ----------
        fingerprints : ndarray ``(N, fp_dim)``
        target_embeddings : ndarray ``(N, target_dim)``
        real_affinities : ndarray ``(N,)``
        epochs : int
            Recalibration training epochs.

        Returns
        -------
        float
            Final loss after recalibration.
        """
        for i in range(len(real_affinities)):
            self.add_observation(
                fingerprints[i], target_embeddings[i], float(real_affinities[i]),
            )

        logger.info(
            "Recalibrating surrogate with %d new observations (total buffer: %d).",
            len(real_affinities), self.buffer_size,
        )
        return self.fit(epochs=epochs)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save model weights and buffer to disk."""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "buffer_fps": self._buffer_fps,
                "buffer_targets": self._buffer_targets,
                "buffer_affinities": self._buffer_affinities,
                "is_trained": self._is_trained,
                "fp_dim": self.fp_dim,
                "target_dim": self.target_dim,
            },
            path,
        )
        logger.info("Surrogate saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights and buffer from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self._buffer_fps = checkpoint["buffer_fps"]
        self._buffer_targets = checkpoint["buffer_targets"]
        self._buffer_affinities = checkpoint["buffer_affinities"]
        self._is_trained = checkpoint["is_trained"]
        self.model.eval()
        logger.info(
            "Surrogate loaded from %s (buffer: %d, trained: %s)",
            path, self.buffer_size, self._is_trained,
        )
