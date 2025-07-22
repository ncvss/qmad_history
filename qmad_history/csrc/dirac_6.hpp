#include <torch/extension.h>

namespace qmad_history {

at::Tensor dw_templ_mtsg_tmgsMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, double mass);
                            
at::Tensor dw_templ_mtsg_tmgsMhs_backw (const at::Tensor& U_tensor, const at::Tensor& grad_tensor,
                                  const at::Tensor& hops_tensor, double mass);

at::Tensor dw_templ_mtsg_tmsgMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, double mass);

at::Tensor dw_templbound_mtsg_tmsgMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, const at::Tensor& bound_tensor, double mass);


at::Tensor dwc_templ_mtsg_tmgsMhns (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                    const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                                    double mass, double csw);

at::Tensor dwc_templ_mtsg_tmsgMhns (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                    const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                                    double mass, double csw);

at::Tensor dw_roof_templ_mtsg_tmgsMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                       const at::Tensor& hops_tensor, double mass);

at::Tensor dw_tempipe_mtsg_tmgsMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, double mass);

}
