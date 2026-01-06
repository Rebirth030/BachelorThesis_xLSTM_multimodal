"""Model training and inference utilities for LSTM-based architectures."""

import os
import random
import time
import json
import copy
from typing import Optional, Literal, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from xLSTM import xLSTM
from LSTM import LSTM

from data_prep import PrepAndDataLoader


class Timer:
    """Simple wall-clock timer utility."""

    def __init__(self):
        """Initialize the timer without starting it."""
        self._t0 = None

    def start(self):
        """Start the timer."""
        self._t0 = time.time()

    def stop(self, prefix: str = "[Timer] Elapsed"):
        """Stop the timer and print the elapsed time.

        Args:
            prefix: Message prefix used for the printed output.
        """
        if self._t0 is None:
            print("[Timer] Not started.")
            return
        dt = time.time() - self._t0
        print(f"{prefix}: {dt:.2f}s")
        self._t0 = None

def set_global_seed(seed: int = 42, strict_cudnn: bool = False):
    """Set random seeds for NumPy, Python, and PyTorch.

    Args:
        seed: Seed value used across RNGs.
        strict_cudnn: Whether to enable deterministic CuDNN behavior.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Model:
    """Train, validate, and run inference for direct multi-step models."""

    def __init__(self):
        """Initialize model metadata, optimizer, and cached data containers."""
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type: Literal["xLSTM", "LSTM"] = "xLSTM"
        self._optimizer: Optional[optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None

        self.window_size: Optional[int] = None
        self.n_features: Optional[int] = None
        self.units: Optional[int] = None
        self.lstm_layers: Optional[int] = None
        self.horizon: Optional[int] = None
        self.dropout: Optional[float] = None

        self._cache: Dict[str, Any] = {
            "X_tr": None, "y_tr": None,
            "X_va": None, "y_va": None,
            "X_te": None, "y_te": None,
            "base_tr": None, "baseT_tr": None,
            "base_va": None, "baseT_va": None,
            "base_te": None, "baseT_te": None,
            "normalise": None, "norm_method": None
        }

    def build(
        self,
        config: Dict[str, Any],
        model_type: Literal["xLSTM", "LSTM"] = "xLSTM",
    ) -> None:
        """Build the model architecture and optimizer from a configuration dict.

        Args:
            config: Configuration containing architecture and optimizer fields.
            model_type: Selects between "xLSTM" and "LSTM".

        Returns:
            None.
        """
        self.model_type = model_type
        self.window_size = int(config["window_size"])
        self.n_features = int(config["n_features"])
        self.horizon = int(config["horizon"])
        self.units = int(config["units"])
        self.dropout = float(config["dropout"])

        optimizer = config["optimizer"]
        learning_rate = float(config["learning_rate"])
        loss = config["loss"]

        if model_type == "xLSTM":
            self.model = xLSTM(
                window_size=self.window_size,
                n_features=self.n_features,
                horizon=self.horizon,
                dropout=self.dropout,
                num_blocks=int(config["xlstm_num_blocks"]),
                slstm_at=config["xlstm_slstm_at"],
                hidden_size=self.units,
                mlstm_conv1d_kernel_size=config["xlstm_mlstm_conv1d_kernel_size"],
                mlstm_qkv_proj_blocksize=config["xlstm_mlstm_qkv_proj_blocksize"],
                mlstm_num_heads=config["xlstm_mlstm_num_heads"],
                mlstm_proj_factor=config["xlstm_mlstm_proj_factor"],
                slstm_conv1d_kernel=config["xlstm_slstm_conv1d_kernel"],
                slstm_num_heads=config["xlstm_slstm_num_heads"],
            ).to(self.device)
        elif model_type == "LSTM":
            self.lstm_layers = int(config["lstm_layers"])
            self.model = LSTM(
                n_features=self.n_features,
                hidden_size=self.units,
                num_layers=self.lstm_layers,
                horizon=self.horizon,
                dropout=self.dropout,
            ).to(self.device)
        else:
            raise ValueError(f"Model type {model_type} not supported")

        if optimizer == "adam":
            self._optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "adamW":
            self._optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            self._optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        if loss != "mse":
            raise ValueError("Only 'mse' is supported as loss.")
        self._criterion = nn.MSELoss()

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] {model_type} compiled:")
        print(f"  Input: seq_len={self.window_size}, features={self.n_features} | hidden={self.units} | horizon={self.horizon}")
        if model_type == "xLSTM":
            print(f"  xLSTM blocks={config['xlstm_num_blocks']}, sLSTM at={config['xlstm_slstm_at']}")
        if model_type == "LSTM":
            print(f"  LSTM layers={self.lstm_layers}")
        print(f"  Trainable params: {n_params:,}")

    def prepare_data_from_prep(
        self,
        prep: "PrepAndDataLoader",
        *,
        normalise: bool,
        window_size: int,
        prediction_range: int,
        norm_method: Literal["percentage", "minmax"] = "percentage",
        stride: int = 1,
        verbose: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create train/validation/test windows from a PrepAndDataLoader instance.

        Args:
            prep: Prepared data source with split accessors.
            normalise: Whether to normalize the windows.
            window_size: Number of past steps per input window.
            prediction_range: Number of steps predicted per window.
            norm_method: Normalization method to apply.
            stride: Step size between windows.
            verbose: Verbosity level for shape logging.

        Returns:
            Tuple of X/y arrays for train, validation, and test splits.
        """
        X_tr, y_tr, b_tr, bT_tr = prep.get_train_data(
            normalise=normalise, window_size=window_size,
            prediction_range=prediction_range, norm_method=norm_method, stride=stride
        )
        X_va, y_va, b_va, bT_va = prep.get_validation_data(
            normalise=normalise, window_size=window_size,
            prediction_range=prediction_range, norm_method=norm_method, stride=stride
        )
        X_te, y_te, b_te, bT_te = prep.get_test_data(
            normalise=normalise, window_size=window_size,
            prediction_range=prediction_range, norm_method=norm_method, stride=stride
        )

        self._cache.update({
            "X_tr": X_tr, "y_tr": y_tr,
            "X_va": X_va, "y_va": y_va,
            "X_te": X_te, "y_te": y_te,
            "base_tr": b_tr, "baseT_tr": bT_tr,
            "base_va": b_va, "baseT_va": bT_va,
            "base_te": b_te, "baseT_te": bT_te,
            "normalise": normalise, "norm_method": norm_method
        })

        if verbose:
            print("[Data] Shapes ->",
                  "X_train", X_tr.shape, "y_train", y_tr.shape,
                  "| X_val", X_va.shape, "y_val", y_va.shape,
                  "| X_test", X_te.shape, "y_test", y_te.shape)

        return X_tr, y_tr, X_va, y_va, X_te, y_te

    @staticmethod
    def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        """Create a PyTorch DataLoader for feature/target arrays.

        Args:
            X: Input array shaped [N, S, F].
            y: Target array shaped [N, H] or [N, H, 1].
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the dataset.

        Returns:
            A DataLoader over the given dataset.
        """
        if y.ndim == 3 and y.shape[-1] == 1:
            y = y.squeeze(-1)

        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        cpu_workers = 4

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=cpu_workers,
            persistent_workers=(cpu_workers > 0),
            prefetch_factor=2 if cpu_workers > 0 else None,
        )

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        *,
        epochs: int = 100,
        batch_size: int = 128,
        save_dir: str = "./checkpoints",
        patience: int = 10,
        monitor: Literal["val_loss"] = "val_loss",
        verbose: int = 1,
        restore_best_weights: bool = True,
        save_best_only: bool = True,
    ) -> str:
        """Train the model with NumPy arrays and save checkpoints.

        Args:
            x_train: Training inputs shaped [N, S, F].
            y_train: Training targets shaped [N, H] or [N, H, 1].
            x_val: Validation inputs shaped [N, S, F].
            y_val: Validation targets shaped [N, H] or [N, H, 1].
            epochs: Number of epochs to train.
            batch_size: Batch size for training.
            save_dir: Directory for checkpoints and history.
            patience: Early-stopping patience based on validation loss.
            monitor: Metric name to monitor.
            verbose: Verbosity level for progress logging.
            restore_best_weights: Whether to reload the best weights into memory.
            save_best_only: Whether to save only the best checkpoint.

        Returns:
            Path to the best checkpoint file.
        """
        assert self.model is not None, "Call build(...) before train(...)."
        assert monitor == "val_loss", "Only 'val_loss' is supported as monitor."
        os.makedirs(save_dir, exist_ok=True)

        train_loader = self._make_loader(x_train, y_train, batch_size, shuffle=True)
        val_loader = self._make_loader(x_val, y_val, batch_size, shuffle=False)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag_layers = (self.lstm_layers if self.model_type == "LSTM" else "x")
        fname = f"{self.model_type}_H{self.horizon}_{timestamp}_b{batch_size}_un{self.units}_lay{tag_layers}_e{epochs}"
        best_path = os.path.join(save_dir, fname + ".pt")
        hist_path = os.path.join(save_dir, fname + "_history.json")

        best_val = float("inf")
        best_state = None
        wait = 0
        history: Dict[str, Any] = {"loss": [], "val_loss": []}

        timer = Timer()
        timer.start()
        if verbose:
            print(f"[Model] Training start | epochs={epochs} | batch_size={batch_size} | horizon={self.horizon}")

        for ep in range(1, epochs + 1):
            self.model.train()
            train_loss_sum, train_n = 0.0, 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self._optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = self._criterion(pred, yb)
                loss.backward()
                self._optimizer.step()
                bs = xb.size(0)
                train_loss_sum += loss.item() * bs
                train_n += bs
            train_loss = train_loss_sum / max(train_n, 1)

            self.model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)
                    loss = self._criterion(pred, yb)
                    bs = xb.size(0)
                    val_loss_sum += loss.item() * bs
                    val_n += bs
            val_loss = val_loss_sum / max(val_n, 1)

            history["loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

            if verbose:
                print(f"Epoch {ep:03d}/{epochs} - loss={train_loss:.6f} - val_loss={val_loss:.6f}")

            improved = val_loss < best_val - 1e-12
            if improved:
                best_val = val_loss
                wait = 0
                best_state = copy.deepcopy(self.model.state_dict())
                payload = {
                    "model_state": best_state,
                    "config": {
                        "window_size": self.window_size,
                        "n_features": self.n_features,
                        "horizon": self.horizon,
                        "units": self.units,
                        "lstm_layers": self.lstm_layers if self.model_type == "LSTM" else None,
                        "dropout": getattr(self.model, "dropout", None),
                        "model_type": self.model_type,
                    },
                    "epoch": ep,
                    "val_loss": float(best_val),
                }
                torch.save(payload, best_path)
                if verbose:
                    print(f"[Checkpoint] Saved best model to: {best_path} (val_loss={best_val:.6f})")
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"[EarlyStopping] No improvement for {patience} epochs. Stopping.")
                    break

        timer.stop("[Model] Training finished")

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        if verbose:
            print(f"[Model] Training history saved to: {hist_path}")
            print(f"[Model] Best checkpoint: {best_path} | best val_loss={best_val:.6f}")

        if restore_best_weights and best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            self.model.eval()
            if verbose:
                print("[Model] Restored best weights into RAM.")

        if not save_best_only:
            last_path = os.path.join(save_dir, fname + "_last.pt")
            torch.save({"model_state": self.model.state_dict()}, last_path)
            if verbose:
                print(f"[Model] Saved last-epoch weights to: {last_path}")

        return best_path

    def load(self, model_path: str):
        """Load model weights from a checkpoint file.

        Args:
            model_path: Path to a checkpoint file.

        Returns:
            None.
        """
        assert self.model is not None, "Call build(...) first to set architecture."
        ckpt = torch.load(model_path, map_location=self.device)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"[Model] Loaded weights from: {model_path}")

    def predict_multi_horizon(
            self,
            x: np.ndarray,
            batch_size: int = 256,
            verbose: int = 0
    ) -> np.ndarray:
        """Run batched inference for multi-horizon outputs.

        Args:
            x: Input array shaped [N, S, F].
            batch_size: Batch size for prediction.
            verbose: Verbosity level for shape logging.

        Returns:
            Array of predictions shaped [N, H].
        """
        assert self.model is not None, "Call build(...) and train(...) or load(...) before prediction."
        ds = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self.device)
                yb = self.model(xb)
                preds.append(yb.cpu().numpy())
        out = np.concatenate(preds, axis=0)
        if verbose:
            print(f"[Predict] {out.shape}")
        return out
