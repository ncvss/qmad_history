import torch

def capab (input_str: str = ""):
    """
    Returns whether the extension code was compiled with parallelisation ("parallelise"),
    vectorisation ("vectorise") or manual CUDA error handling ("cuda_error_handling").

    Returns the corresponding boolean value if the argument is one of the above strings,
    or else a dictionary with all boolean values.
    """
    dummy = torch.ones(1)
    capab_vector = torch.ops.qmad_history.capability_function(dummy)
    capab_dict = {"vectorise": bool(capab_vector[0]), "parallelise": bool(capab_vector[1]), "cuda_error_handling": bool(capab_vector[2])}
    if input_str in capab_dict.keys():
        return capab_dict[input_str]
    else:
        return capab_dict

