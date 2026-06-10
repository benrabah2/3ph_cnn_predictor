import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from model_utils import predict_3phase_batched
from data_utils import Convert_2theta_to_q, Shape_Correction_Function
from config import (
    nb_of_points_per_profile,
    Convert_to_q,
    shape_correction,
    model_name,
    Q_MIN,
    Q_MAX,
    plot_data,
    batch_size,
    output_dir_name,
    output_name_template,
)


def _load_profile(file_name, wave_length):
    data = np.loadtxt(file_name)[:, :2]
    if Convert_to_q:
        if wave_length is None:
            raise ValueError(
                "Wavelength is required when converting 2-theta to q. "
                "Use -w/--wavelength or pass --no-convert-to-q."
            )
        data = Convert_2theta_to_q(data, wave_length, Q_MIN, Q_MAX)
    if shape_correction:
        data = Shape_Correction_Function(data, nb_of_points_per_profile)
    return data


def _extract_index_from_filename(fname, dataset_name=None):
    if not fname.lower().endswith(".dat"):
        return None
    if dataset_name:
        if not fname.startswith(dataset_name):
            return None
        remainder = fname[len(dataset_name):]
        match = re.match(r"\D+(\d+)\.dat$", remainder)
    else:
        match = re.search(r"(\d+)\.dat$", fname)
    if not match:
        return None
    return int(match.group(1))


def _list_candidate_files(folder_name, dataset_name):
    candidates = []
    for fname in os.listdir(folder_name):
        idx = _extract_index_from_filename(fname, dataset_name=dataset_name)
        if idx is None:
            continue
        candidates.append((idx, fname))
    return candidates


def _collect_profiles(folder_name, dataset_name, start, end, wave_length):
    profiles = []
    file_numbers = []

    candidates = _list_candidate_files(folder_name, dataset_name)
    if not candidates:
        return profiles, file_numbers

    all_indices = [idx for idx, _ in candidates]
    if start is None:
        start = min(all_indices)
    if end is None:
        end = max(all_indices)

    for idx, fname in sorted(candidates, key=lambda x: x[0]):
        if idx < start or idx > end:
            continue
        file_name = os.path.join(folder_name, fname)
        print(file_name)
        if not os.path.isfile(file_name):
            continue
        profiles.append(_load_profile(file_name, wave_length))
        file_numbers.append(float(idx))

    return profiles, file_numbers


def _resolve_output_path(data_folder, dataset_name, out_dir=None, out_name=None, out_path=None):
    if out_path:
        return out_path
    if out_dir:
        base_dir = out_dir if os.path.isabs(out_dir) else os.path.join(data_folder, out_dir)
    else:
        base_dir = os.path.join(data_folder, output_dir_name)
    os.makedirs(base_dir, exist_ok=True)
    filename = out_name if out_name else output_name_template.format(dataset=dataset_name)
    return os.path.join(base_dir, filename)


def _save_predictions(
    data_folder, dataset_name, file_numbers, f_bcc, f_fcc, f_cem, wave_length,
    out_dir=None, out_name=None, out_path=None,
):
    output_file = _resolve_output_path(
        data_folder, dataset_name, out_dir=out_dir, out_name=out_name, out_path=out_path
    )
    with open(output_file, 'w') as f:
        f.write("# 3-phase fraction prediction (BCC / FCC / Cementite)\n")
        f.write(f"# model_name: {model_name}\n")
        f.write(f"# convert_to_q: {Convert_to_q}\n")
        f.write(f"# q_range: {Q_MIN} to {Q_MAX}\n")
        f.write(f"# wavelength: {wave_length}\n")
        f.write(f"# timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("File Number\tBCC (%)\tFCC (%)\tCementite (%)\n")
        for fn, bcc, fcc, cem in zip(file_numbers, f_bcc, f_fcc, f_cem):
            f.write(f"{fn}\t{bcc:.4f}\t{fcc:.4f}\t{cem:.6f}\n")
    print(f"Predicted phase fractions saved to {output_file}")


def main(data_folder, name, start, end, wave_length, out_dir=None, out_name=None, out_path=None):
    if name:
        print(name)

    profiles, file_numbers = _collect_profiles(data_folder, name, start, end, wave_length)
    if not profiles:
        dataset_label = name if name else "any dataset"
        raise ValueError(f"No data files found in {data_folder} for {dataset_label}.")

    X = np.stack(profiles)
    if file_numbers:
        print(f"Loaded {len(file_numbers)} profiles (index {int(file_numbers[0])} to {int(file_numbers[-1])}).")

    f_bcc, f_fcc, f_cem = predict_3phase_batched(
        X,
        model_name,
        batch_size=batch_size,
        q_min=Q_MIN,
        q_max=Q_MAX,
    )

    if plot_data:
        fig, ax = plt.subplots()
        ax.plot(file_numbers, f_bcc, '-.', label='BCC',        color='steelblue',  markersize=6)
        ax.plot(file_numbers, f_fcc, '-.', label='FCC',        color='darkorange', markersize=6)
        ax.plot(file_numbers, f_cem, '-.', label='Cementite',  color='green',      markersize=6)
        ax.set_ylabel('Predicted phase fraction (%)', fontsize=12)
        ax.set_xlabel('File number', fontsize=12)
        ax.legend()
        plt.show()

    dataset_label = name if name else os.path.basename(os.path.abspath(data_folder))
    _save_predictions(
        data_folder, dataset_label, file_numbers, f_bcc, f_fcc, f_cem, wave_length,
        out_dir=out_dir, out_name=out_name, out_path=out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict BCC, FCC, and cementite phase fractions from HEXRD profiles.'
    )
    parser.add_argument('data_folder', type=str, help='Directory containing the data files')
    parser.add_argument('-n', '--name',       type=str,   required=False, help='Dataset name used for file prefix (optional)')
    parser.add_argument('-w', '--wavelength', type=float, required=False, help='Wavelength (required if converting 2-theta to q)')
    parser.add_argument('-s', '--start',      type=int,   required=False, help='Start index (optional; uses min found if omitted)')
    parser.add_argument('-e', '--end',        type=int,   required=False, help='End index (optional; uses max found if omitted)')
    parser.add_argument('--q-min',            type=float, default=Q_MIN,  help='Minimum q value (default from config)')
    parser.add_argument('--q-max',            type=float, default=Q_MAX,  help='Maximum q value (default from config)')
    parser.add_argument('--no-convert-to-q',     action='store_true', help='Disable 2-theta to q conversion')
    parser.add_argument('--no-shape-correction', action='store_true', help='Disable shape correction')
    parser.add_argument('--nb-points',  type=int, default=nb_of_points_per_profile, help='Points per profile (default from config)')
    parser.add_argument('--plot',       action='store_true', help='Plot predictions')
    parser.add_argument('--batch-size', type=int, default=batch_size, help='Batch size for prediction')
    parser.add_argument('--out-dir',  type=str, default=None, help='Output directory (relative to data_folder if not absolute)')
    parser.add_argument('--out-name', type=str, default=None, help='Output filename (overrides template)')
    parser.add_argument('--out-path', type=str, default=None, help='Full output file path (highest priority)')
    args = parser.parse_args()

    Q_MIN                  = args.q_min
    Q_MAX                  = args.q_max
    Convert_to_q           = not args.no_convert_to_q
    shape_correction       = not args.no_shape_correction
    nb_of_points_per_profile = args.nb_points
    plot_data              = args.plot
    batch_size             = args.batch_size

    if Convert_to_q and args.wavelength is None:
        raise ValueError(
            "Wavelength is required when converting 2-theta to q. "
            "Use -w/--wavelength or pass --no-convert-to-q."
        )

    main(
        args.data_folder,
        name=args.name,
        start=args.start,
        end=args.end,
        wave_length=args.wavelength,
        out_dir=args.out_dir,
        out_name=args.out_name,
        out_path=args.out_path,
    )
