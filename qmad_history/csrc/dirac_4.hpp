#include <torch/extension.h>

namespace qmad_history {

at::Tensor dw_eo_pmtsg_pxtMmghs (const at::Tensor& Ue, const at::Tensor& Uo,
                                 const at::Tensor& ve, const at::Tensor& vo,
                                 double mass, std::vector<int64_t> eodim);

at::Tensor dw_eo_hop_pmtsg_ptMmsgh (const at::Tensor& Ue_ten, const at::Tensor& Uo_ten,
                                    const at::Tensor& ve_ten, const at::Tensor& vo_ten,
                                    const at::Tensor& hopse_ten, const at::Tensor& hopso_ten,
                                    double mass);

at::Tensor dw_eo_hop_pmtsg_pMtmsgh (const at::Tensor& Ue_ten, const at::Tensor& Uo_ten,
                                    const at::Tensor& ve_ten, const at::Tensor& vo_ten,
                                    const at::Tensor& hopse_ten, const at::Tensor& hopso_ten,
                                    double mass);

}
