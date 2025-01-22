#include <torch/extension.h>


namespace qmad_history {

// blocked Wilson with loops mu_sign(bx(by(bz(bt(x(y(z(t(mu(g(g_sum(s))))))))))))
at::Tensor dw_block_mxtsg_dbxtsghm (const at::Tensor& U, const at::Tensor& v, double mass, int64_t bls);

// blocked Wilson with loops bx(by(bz(bt(x(y(z(t(mu(g(g_sum(s)))))))))))
at::Tensor dw_block_mxtsg_bxtsghm (const at::Tensor& U, const at::Tensor& v, double mass, int64_t bls);

}
