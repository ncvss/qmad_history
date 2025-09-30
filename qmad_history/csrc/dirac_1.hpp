#include <torch/extension.h>
#include <vector>


namespace qmad_history {

// Wilson with loops x(y(z(t(s(g(mass, g sum(mu)))))))
at::Tensor dw_dir_mxtsg_xtsgMhm (const at::Tensor& U, const at::Tensor& v, double mass);

// Wilson with loops x(y(z(t(mass, mu(g(g sum(s)))))))
at::Tensor dw_dir_mxtsg_xtMmghs (const at::Tensor& U, const at::Tensor& v, double mass);

// Wilson with loops x(y(z(t(mass, mu(mu sign(g(g sum(s))))))))
at::Tensor dw_dir_mxtsg_xtMmdghs (const at::Tensor& U, const at::Tensor& v, double mass);

// Wilson Clover that computes field strength with 4 loops
// it is false because it only has 1 clover term and is still slow
at::Tensor dwc_dir_mxtsg_false (const at::Tensor& U, const at::Tensor& v, double mass, double csw);

// Wilson Clover with loops x(y(z(t(s(g(g sum(mu, munu)))))))
// field strength terms are precomputed as F_munu
at::Tensor dwc_fpre_mntsg_xtsghmn (const at::Tensor& U, const at::Tensor& v,
                                   const std::vector<at::Tensor>& F,
                                   double mass, double csw);

// Wilson Clover with loops x(y(z(t(mu(g(g sum(s,munu)))))))
// field strength terms are precomputed as F_munu
at::Tensor dwc_fpre_mntsg_xtmghsn (const at::Tensor& U, const at::Tensor& v,
                                   const std::vector<at::Tensor>& F,
                                   double mass, double csw);

// Wilson Clover with loops x(y(z(t(munu,mu(g(g sum(s)))))))
// field strength terms are precomputed as F_munu
at::Tensor dwc_fpre_mntsg_xtmnghs (const at::Tensor& U, const at::Tensor& v,
                                   const std::vector<at::Tensor>& F,
                                   double mass, double csw);

// Wilson Clover with loops x(y(z(t(munu,mu(mu sign(g(g sum(s))))))))
// field strength terms are precomputed as F_munu
at::Tensor dwc_fpre_mntsg_xtmdnghs (const at::Tensor& U, const at::Tensor& v,
                                    const std::vector<at::Tensor>& F,
                                    double mass, double csw);



}