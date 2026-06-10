Convert_to_q = True
shape_correction = True
nb_of_points_per_profile = 1500

Q_MIN = 2.0
Q_MAX = 8.0
CEM_MAX = 0.1
# 5 crystallographically validated cementite windows (index ranges in the 1500-pt q grid)
# Win1[2.50-2.87] Win2[3.20-3.33] Win3[3.56-4.19] Win4[5.00-5.15] Win5[5.47-5.57]
CEM_WINDOWS = [(125, 219), (300, 333), (390, 546), (750, 787), (867, 892)]

plot_data = False
batch_size = 512
output_dir_name = "results"
output_name_template = "predicted_phase_fractions_{dataset}.txt"

model_name = 'model_3ph_v1.0.0.keras'
