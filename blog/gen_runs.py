#!/usr/bin/env python3

print("rm tune_with_ax.out")
print("rm tune_with_ax.err")

# runtime = 30 * 60  # 30min
runtime = 2 * 60 * 60  # 2hr

# for mode, num_models in [('siso', [1]),
#                          ('simo', [2, 4, 8]),
#                          ('simo_ld', [2, 4, 8]),
#                          ('mimo', [2, 4, 8])]:
#     for m in num_models:
#         print("python3 tune_with_ax.py"
#               " --group 11_full_rerun"
#               f" --mode {mode}"
#               f" --num-models {m}"
#               f" --run-time-sec {runtime}"
#               " --epochs 50"
#               " >>tune_with_ax.out"
#               " 2>>tune_with_ax.err")

for mode in ['siso', 'simo', 'simo_ld', 'mimo']:
    print("python3 tune_with_ax.py"
          " --group 12_full_rerun"
          f" --mode {mode}"
          f" --num-models 4"
          f" --run-time-sec {runtime}"
          " --epochs 60"
          " >>tune_with_ax.out"
          " 2>>tune_with_ax.err")
