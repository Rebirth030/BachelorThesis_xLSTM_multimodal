"""Data preparation utilities for time-series model training."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple, Literal

from sklearn.preprocessing import MinMaxScaler


class PrepAndDataLoader:
    """Prepare time-series data for LSTM-based models."""

    def __init__(
        self,
        filename: str,
        training_split: float,
        validation_split: float,
        cols: List[str],
        target_col: Union[str, int] = 0,
        dtype: str = "float32",
        transform_batch_size: Optional[int] = 200_000,
        eps: float = 1e-8,
    ):
        """Load, split, and configure time-series datasets.

        Args:
            filename: CSV file path with a Date column.
            training_split: Fraction of data used for training.
            validation_split: Fraction of data used for validation.
            cols: Feature column names to load.
            target_col: Target column name or index within cols.
            dtype: NumPy dtype for stored arrays.
            transform_batch_size: Placeholder for API compatibility.
            eps: Small constant for numerical stability.
        """
        self.dtype = dtype
        self.feature_cols = cols
        self.transform_batch_size = transform_batch_size
        self.eps = float(eps)

        if not (0.0 < training_split < 1.0):
            raise ValueError("training_split must be in (0, 1).")
        if not (0.0 <= validation_split < 1.0):
            raise ValueError("validation_split must be in [0, 1).")
        if training_split + validation_split >= 1.0:
            raise ValueError("training_split + validation_split must be < 1.0.")

        raw_df = pd.read_csv(filename, parse_dates=["Date"])
        raw_df = raw_df.sort_values("Date").set_index("Date")

        df = raw_df[self.feature_cols]
        if df.empty:
            raise ValueError("Loaded DataFrame is empty. Please check file and columns.")

        n = len(df)
        n_train = int(n * training_split)
        n_val = int(n * validation_split)

        if n_train <= 0 or (n_train + n_val) >= n:
            raise ValueError(
                "Invalid splits, one of train/validation/test would be empty: "
                f"n={n}, n_train={n_train}, n_val={n_val}"
            )

        self.data_train = df.iloc[:n_train]
        self.data_validation = df.iloc[n_train:n_train + n_val]
        self.data_test = df.iloc[n_train + n_val:]

        if isinstance(target_col, str):
            if target_col not in self.feature_cols:
                raise ValueError(f"target_col '{target_col}' not found in {self.feature_cols}")
            self.target_idx = self.feature_cols.index(target_col)
        else:
            self.target_idx = int(target_col)

        if not (0 <= self.target_idx < len(self.feature_cols)):
            raise IndexError(
                f"target_col index {self.target_idx} out of range "
                f"[0, {len(self.feature_cols) - 1}] for cols={self.feature_cols}"
            )

        target_name = self.feature_cols[self.target_idx]
        self.target_scaler = MinMaxScaler()
        self.target_scaler.fit(
            df[[target_name]].to_numpy(copy=False).astype(self.dtype)
        )

    def _prepare_windows(
        self,
        data: pd.DataFrame,
        window_size: int,
        prediction_range: int = 5,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sliding windows and corresponding base values.

        Args:
            data: DataFrame slice for a specific split.
            window_size: Number of past steps per input window.
            prediction_range: Number of steps to predict per window.
            stride: Step size between windows.

        Returns:
            Tuple of (data_windows, base_values, base_target_values).
        """
        if window_size <= 0 or prediction_range <= 0:
            raise ValueError("window_size and prediction_range must be > 0.")
        if stride <= 0:
            raise ValueError("stride must be > 0.")

        arr = data.to_numpy(copy=False).astype(self.dtype)
        T, F = arr.shape
        total = window_size + prediction_range

        if T < total:
            raise ValueError(
                f"Split has too few rows ({T}) for window_size+prediction_range={total}."
            )

        data_windows = []
        base_values = []
        base_target_values = []

        for i in range(0, T - total + 1, stride):
            window = arr[i:i + total]
            data_windows.append(window)
            base_values.append(window[0])
            base_target_values.append([window[0, self.target_idx]])

        data_windows = np.asarray(data_windows, dtype=self.dtype)
        base_values = np.asarray(base_values, dtype=self.dtype)
        base_target_values = np.asarray(base_target_values, dtype=self.dtype)
        return data_windows, base_values, base_target_values

    def _pick_split_df(self, split: Literal["train", "validation", "test"]) -> pd.DataFrame:
        """Select the DataFrame slice for a named split.

        Args:
            split: Split name ("train", "validation", or "test").

        Returns:
            DataFrame for the requested split.
        """
        if split == "train":
            return self.data_train
        if split == "validation":
            return self.data_validation
        if split == "test":
            return self.data_test
        raise ValueError(f"Unknown split: {split}")

    def _pick_dates(self, split: Literal["train", "validation", "test"]) -> np.ndarray:
        """Return the index dates for a named split.

        Args:
            split: Split name ("train", "validation", or "test").

        Returns:
            NumPy array of datetime values.
        """
        df = self._pick_split_df(split)
        return df.index.to_numpy()

    def _get_split(
        self,
        split: Literal["train", "validation", "test"],
        *,
        normalise: bool,
        window_size: int,
        prediction_range: int = 5,
        norm_method: Literal["percentage", "minmax"] = "percentage",
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build input and target windows for a specific split.

        Args:
            split: Split name ("train", "validation", or "test").
            normalise: Whether to normalize the windows.
            window_size: Number of past steps per input window.
            prediction_range: Number of steps to predict per window.
            norm_method: Normalization method to apply.
            stride: Step size between windows.

        Returns:
            Tuple of (X, y, base_values, base_target_values).
        """
        df = self._pick_split_df(split)
        windows, base_values, base_target_values = self._prepare_windows(
            df, window_size, prediction_range, stride
        )

        if normalise:
            windows = self._normalise_windows(windows, method=norm_method)

        X = windows[:, :window_size, :]
        y = windows[:, window_size:, [self.target_idx]]

        return (
            X.astype(self.dtype, copy=False),
            y.astype(self.dtype, copy=False),
            base_values.astype(self.dtype, copy=False),
            base_target_values.astype(self.dtype, copy=False),
        )

    def get_train_data(self, *args, **kwargs):
        """Return training split windows and base values.

        Args:
            *args: Positional arguments forwarded to _get_split.
            **kwargs: Keyword arguments forwarded to _get_split.

        Returns:
            Tuple of (X, y, base_values, base_target_values).
        """
        return self._get_split("train", *args, **kwargs)

    def get_validation_data(self, *args, **kwargs):
        """Return validation split windows and base values.

        Args:
            *args: Positional arguments forwarded to _get_split.
            **kwargs: Keyword arguments forwarded to _get_split.

        Returns:
            Tuple of (X, y, base_values, base_target_values).
        """
        return self._get_split("validation", *args, **kwargs)

    def get_test_data(self, *args, **kwargs):
        """Return test split windows and base values.

        Args:
            *args: Positional arguments forwarded to _get_split.
            **kwargs: Keyword arguments forwarded to _get_split.

        Returns:
            Tuple of (X, y, base_values, base_target_values).
        """
        return self._get_split("test", *args, **kwargs)

    def get_prediction_dates(
        self,
        split: Literal["train", "validation", "test"],
        *,
        window_size: int,
        prediction_range: int = 5,
        stride: int = 1,
    ) -> np.ndarray:
        """Return target window dates aligned with prediction horizons.

        Args:
            split: Split name ("train", "validation", or "test").
            window_size: Number of past steps per input window.
            prediction_range: Number of steps predicted per window.
            stride: Step size between windows.

        Returns:
            Array of prediction dates shaped [N, prediction_range].
        """
        dates = self._pick_dates(split)
        T = len(dates)
        total = window_size + prediction_range

        if window_size <= 0 or prediction_range <= 0:
            raise ValueError("window_size and prediction_range must be > 0.")
        if stride <= 0:
            raise ValueError("stride must be > 0.")
        if T < total:
            raise ValueError(
                f"Split has too few rows ({T}) for window_size+prediction_range={total}."
            )

        pred_dates = []
        for i in range(0, T - total + 1, stride):
            pred_dates.append(dates[i + window_size : i + total])

        return np.stack(pred_dates, axis=0)

    def _normalise_windows(
        self,
        window_data: np.ndarray,
        method: Literal["percentage", "minmax"] = "percentage",
    ) -> np.ndarray:
        """Normalize windows along the target dimension.

        Args:
            window_data: Array shaped [N, S, F].
            method: Normalization method to apply.

        Returns:
            Normalized windows with the same shape as input.
        """
        normed = window_data.astype(self.dtype, copy=True)
        N, S, F = normed.shape
        t = self.target_idx

        if method == "percentage":
            x0 = normed[:, 0:1, t:t + 1]
            safe = np.where(
                np.abs(x0) < self.eps,
                np.sign(x0) * self.eps + self.eps,
                x0
            ).astype(self.dtype)

            normed[:, :, t:t + 1] = (normed[:, :, t:t + 1] / safe) - 1.0
            return normed

        if method == "minmax":
            target_flat = normed[:, :, t].reshape(-1, 1)
            target_scaled = self.target_scaler.transform(target_flat)
            normed[:, :, t] = target_scaled.reshape(N, S)
            return normed

        raise ValueError(f"Unknown normalization method: {method!r}")

    def denormalise(
        self,
        normalized_data: np.ndarray,
        method: Literal["percentage", "minmax"] = "percentage",
        base_values: Optional[np.ndarray] = None,
        normalise: bool = True,
    ) -> np.ndarray:
        """Reverse normalization for target-only windows.

        Args:
            normalized_data: Target-only array shaped [N, S, 1].
            method: Normalization method previously applied.
            base_values: Base target values shaped [N, 1] for percentage denormalization.
            normalise: Whether normalization was applied in the first place.

        Returns:
            Denormalized target values shaped [N, S, 1].
        """
        if not normalise:
            return normalized_data.astype(self.dtype, copy=False)

        if method == "minmax":
            N, S, C = normalized_data.shape
            if C != 1:
                raise ValueError("minmax denormalise expects target-only data with C == 1.")

            flat = normalized_data.reshape(-1, 1).astype(self.dtype, copy=False)
            inv = self.target_scaler.inverse_transform(flat)
            return inv.reshape(N, S, 1).astype(self.dtype, copy=False)

        if method == "percentage":
            if base_values is None:
                raise ValueError("base_values are required for percentage denormalisation.")

            if base_values.ndim != 2 or base_values.shape[1] != 1:
                raise ValueError("For target-only percentage denormalisation, base_values must have shape (N, 1).")

            safe_base = np.where(
                np.abs(base_values) < self.eps,
                np.sign(base_values) * self.eps + self.eps,
                base_values
            ).astype(self.dtype)
            return ((normalized_data + 1.0) * safe_base[:, None, :]).astype(self.dtype, copy=False)

        raise ValueError(f"Unknown denormalisation method: {method!r}")
