import torch


def _dw_templ_mtsg_backward(ctx, grad):
    U, v, h, mass = ctx.saved_tensors
    grad_v = None
    # currently, we only implement the v gradient
    if ctx.needs_input_grad[1]:
        grad_v = torch.ops.qmad_history.dw_templ_mtsg_tmgsMhs_backw(U, grad, h, mass.item())
    return None, grad_v, None, None, None

def _dw_templ_mtsg_setup_context(ctx, inputs, output):
    U, v, hops, mass = inputs
    saved_U, saved_v, saved_hops, saved_mass = None, None, None, None
    # currently, we only implement the v gradient
    if ctx.needs_input_grad[1]:
        saved_U = U
        saved_mass = torch.tensor([mass]) # apparently this can only save tensors
        saved_hops = hops
    ctx.save_for_backward(saved_U, saved_v, saved_hops, saved_mass)



def _dwc_grid_mtsg_backward(ctx, grad):
    U, v, fs, h, mass = ctx.saved_tensors
    grad_v = None
    # currently, we only implement the v gradient
    if ctx.needs_input_grad[1]:
        grad_v = torch.ops.qmad_history.dwc_grid_mtsg_backw(U, grad, fs, h, mass.item())
    return None, grad_v, None, None, None

def _dwc_grid_mtsg_setup_context(ctx, inputs, output):
    U, v, fs, hops, mass = inputs
    saved_U, saved_v, saved_fs, saved_hops, saved_mass = None, None, None, None, None
    # currently, we only implement the v gradient
    if ctx.needs_input_grad[1]:
        saved_U = U
        saved_fs = fs
        saved_mass = torch.tensor([mass]) # apparently this can only save tensors
        saved_hops = hops
    ctx.save_for_backward(saved_U, saved_v, saved_fs, saved_hops, saved_mass)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd("qmad_history::dwc_grid_mtsg_tmngsMhs",
                                _dwc_grid_mtsg_backward, setup_context=_dwc_grid_mtsg_setup_context)

torch.library.register_autograd("qmad_history::dw_templ_mtsg_tmgsMhs",
                                _dw_templ_mtsg_backward, setup_context=_dw_templ_mtsg_setup_context)

