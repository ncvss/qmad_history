# import os

# this_dir = os.path.dirname(os.path.curdir)
# settings_file = os.path.join(this_dir, "qmad_history", "settings.txt")

# capab = dict()

# # take the settings from the text file
# with open(settings_file, "r") as sfile:
#     data = sfile.readlines()
#     for li in data:
#         wo = li.split()
#         capab[wo[0]] = bool(wo[1])

# # take settings from text file at execution time
# def parallelise():
#     with open(settings_file, "r") as sfile:
#         data = sfile.readlines()
#         for li in data:
#             wo = li.split()
#             if wo[0] == "parallelise":
#                 returnvalue = bool(wo[1])
#     return returnvalue

# def vectorise():
#     with open(settings_file, "r") as sfile:
#         data = sfile.readlines()
#         for li in data:
#             wo = li.split()
#             if wo[0] == "vectorise":
#                 returnvalue = bool(wo[1])
#     return returnvalue

# capab = {"parallelise": True, "vectorise": True, "cuda_error_handling": True}


import torch

def capab (input_str = None):
    dummy = torch.ones(1)
    capab_vector = torch.ops.qmad_history.capability_function(dummy)
    capab_dict = {"vectorise": bool(capab_vector[0]), "parallelise": bool(capab_vector[1]), "cuda_error_handling": bool(capab_vector[2])}
    if input_str in capab_dict.keys():
        return capab_dict[input_str]
    else:
        return capab_dict

