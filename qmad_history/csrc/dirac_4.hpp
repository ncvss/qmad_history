#include <torch/extension.h>

namespace qmad_history {

at::Tensor dw_eo_pmtsg_pxtMmghs (const at::Tensor& Ue, const at::Tensor& Uo,
                                 const at::Tensor& ve, const at::Tensor& vo,
                                 double mass, std::vector<int64_t> eodim);

}
