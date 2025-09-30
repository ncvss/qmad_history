import torch
import qmad_history.settings

# test if the capability function works

print(qmad_history.settings.capab())
for ch in ["vectorise", "parallelise", "cuda_error_handling"]:
    print(ch, qmad_history.settings.capab(ch))
