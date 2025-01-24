#include <torch/extension.h>

namespace qmad_history {

at::Tensor dw_tempdir_mtsg_tmgsMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  double mass);

}
