#include <torch/extension.h>

namespace qmad_history {

at::Tensor dw_templ_mtsg_tmgsMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, double mass);

at::Tensor dwc_templ_mtsg_tmgsMhns (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                    const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                                    double mass, double csw);

}
