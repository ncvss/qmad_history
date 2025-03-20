#include <torch/extension.h>

namespace qmad_history{

at::Tensor dw_templ_mtsgt_tmgsMht (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, double mass);

}
