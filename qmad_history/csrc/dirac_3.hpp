#include <torch/extension.h>

namespace qmad_history {

// memory U[mu,t,g,h] and v[t,s,h]

at::Tensor dw_hop_mtsg_tMmgsh (const at::Tensor& U_ten, const at::Tensor& v_ten,
                               const at::Tensor& hops_ten, double mass);

at::Tensor dw_hop_mtsg_tMgshm (const at::Tensor& U_ten, const at::Tensor& v_ten,
                               const at::Tensor& hops_ten, double mass);

at::Tensor dw_hop_mtsg_tmgsMh (const at::Tensor& U_ten, const at::Tensor& v_ten,
                               const at::Tensor& hops_ten, double mass);

at::Tensor dw_hop_mtsg_tmsgMh_cpu (const at::Tensor& U_ten, const at::Tensor& v_ten,
                               const at::Tensor& hops_ten, double mass);



// memory U[t,mu,g,h] and v[t,h,s]

at::Tensor dw_hop_tmgs_tMmghs (const at::Tensor& U_ten, const at::Tensor& v_ten,
                               const at::Tensor& hops_ten, double mass);

at::Tensor dw_hop_tmgs_tMmgsh (const at::Tensor& U_ten, const at::Tensor& v_ten,
                               const at::Tensor& hops_ten, double mass);

at::Tensor dw_hop_tmgs_tMmgshu (const at::Tensor& U_ten, const at::Tensor& v_ten,
                               const at::Tensor& hops_ten, double mass);

at::Tensor dw_hop_tmgs_tmgsMh (const at::Tensor& U_ten, const at::Tensor& v_ten,
                               const at::Tensor& hops_ten, double mass);



}
