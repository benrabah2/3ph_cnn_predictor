import numpy as np
from scipy.interpolate import interp1d


def _fill_nan_intensities(X_dict_data):
    """Replace NaN intensity values via linear interpolation (e.g. dead detector pixels)."""
    q, I = X_dict_data[:, 0], X_dict_data[:, 1].copy()
    nan_mask = np.isnan(I)
    if nan_mask.any():
        nan_q = q[nan_mask]
        valid = ~nan_mask
        I[nan_mask] = np.interp(nan_q, q[valid], I[valid])
        X_dict_data = X_dict_data.copy()
        X_dict_data[:, 1] = I
        print(f"  [NaN fix] {nan_mask.sum()} NaN intensity value(s) at q={nan_q} filled by linear interpolation.")
    return X_dict_data


def Convert_2theta_to_q(X_dict_data, wave_length, q_min, q_max):
    X_dict_data[:, 0] = np.abs(np.sin(np.radians(X_dict_data[:, 0] / 2)) * 4 * np.pi / wave_length)
    index_q = np.where(np.logical_and(X_dict_data[:, 0] >= q_min, X_dict_data[:, 0] <= q_max))[0]
    return _fill_nan_intensities(X_dict_data[index_q, :])


def Shape_Correction_Function(X_dict_data, nb_of_points_per_profile):
    original_array = X_dict_data[:, 1]
    original_array_x = X_dict_data[:, 0]
    index_shape = np.linspace(np.min(original_array_x), np.max(original_array_x), len(original_array))
    interpolation_func = interp1d(index_shape, original_array, kind='linear')
    new_index_shape = np.linspace(np.min(original_array_x), np.max(original_array_x), nb_of_points_per_profile)
    interpolated_array = interpolation_func(new_index_shape)
    return np.column_stack((new_index_shape, interpolated_array))


def normalize_log_intensity(X, q_min, q_max):
    """
    X shape: (N, N_PTS, 2) — channels [q, I]
    q normalized to [0, 1] using fixed q_min/q_max.
    I is first rescaled per-profile so its own max is 1000 (matches
    normalize_array() in the training data-prep pipeline, build_ml_real_data/
    00_prepare_ml_inputs_3ph.ipynb), then transformed as log1p(I) / log1p(1000).
    Must match the preprocessing used during training of model_3ph_v1.0.0.keras.
    """
    Xn = X.astype(np.float32).copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - q_min) / (q_max - q_min)
    I = Xn[:, :, 1]
    I_max = np.max(I, axis=1, keepdims=True)
    I = np.divide(I, I_max, out=np.zeros_like(I), where=I_max != 0) * 1000.0
    Xn[:, :, 1] = np.log1p(I) / np.log1p(1000.0)
    return Xn


def extract_cem_windows(X, cem_windows):
    """Slice and concatenate cementite q-windows. X: (N, N_PTS, 2) → (N, N_CEM_PTS, 2)."""
    return np.concatenate([X[:, s:e, :] for s, e in cem_windows], axis=1)
