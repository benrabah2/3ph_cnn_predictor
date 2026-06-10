import tensorflow as tf
import numpy as np

from data_utils import normalize_log_intensity, extract_cem_windows
from config import CEM_MAX, CEM_WINDOWS


def load_model(model_name):
    return tf.keras.models.load_model(model_name)


def reconstruct_fractions(cem_present, cem_frac_norm, p_bcc, cem_max=CEM_MAX):
    """
    Cascade reconstruction matching the training target definition:
      cem    = round(cem_present * cem_frac_norm * cem_max, 6)
      f_bcc  = (1 - cem) * p_bcc
      f_fcc  = (1 - cem) * (1 - p_bcc)
    """
    cem_present   = np.clip(cem_present.reshape(-1),   0.0, 1.0)
    cem_frac_norm = np.clip(cem_frac_norm.reshape(-1), 0.0, 1.0)
    p_bcc         = np.clip(p_bcc.reshape(-1),         0.0, 1.0)
    cem   = np.round(cem_present * cem_frac_norm * cem_max, 6)
    rem   = np.clip(1.0 - cem, 0.0, 1.0)
    f_bcc = rem * p_bcc
    f_fcc = rem * (1.0 - p_bcc)
    return f_bcc, f_fcc, cem


def predict_3phase_batched(X_data, model_name, batch_size=512, q_min=2.0, q_max=8.0):
    """
    X_data: (N, N_PTS, 2) — raw [q, I], not yet normalized.
    Returns (f_bcc, f_fcc, f_cem) each of shape (N,).
    Normalization, CEM window extraction, and fraction reconstruction
    are all performed here — call with the raw stacked profiles.
    """
    model = load_model(model_name)
    all_cem_present, all_cem_frac_norm, all_p_bcc = [], [], []

    for start in range(0, len(X_data), batch_size):
        X_batch = X_data[start:start + batch_size]
        X_norm  = normalize_log_intensity(X_batch, q_min, q_max)
        X_cem   = extract_cem_windows(X_norm, CEM_WINDOWS)
        cem_present, cem_frac_norm, p_bcc = model.predict([X_norm, X_cem], verbose=0)
        all_cem_present.append(cem_present)
        all_cem_frac_norm.append(cem_frac_norm)
        all_p_bcc.append(p_bcc)

    f_bcc, f_fcc, f_cem = reconstruct_fractions(
        np.concatenate(all_cem_present),
        np.concatenate(all_cem_frac_norm),
        np.concatenate(all_p_bcc),
    )
    return np.around(f_bcc, 4), np.around(f_fcc, 4), np.around(f_cem, 6)
