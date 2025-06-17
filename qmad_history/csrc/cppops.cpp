#include <torch/extension.h>
#include <vector>

#include "dirac_1.hpp"
#include "dirac_2.hpp"
#include "dirac_3.hpp"
#include "dirac_4.hpp"
#include "dirac_5.hpp"
#include "dirac_6.hpp"
#include "dirac_7.hpp"

#include "dirac_9.hpp"


namespace qmad_history{


// Registers _C as a Python extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(qmad_history, m) {
    m.def("dw_dir_mxtsg_xtsgMhm(Tensor U, Tensor v, float mass) -> Tensor");
    m.def("dw_dir_mxtsg_xtMmghs(Tensor U, Tensor v, float mass) -> Tensor");
    m.def("dw_dir_mxtsg_xtMmdghs(Tensor U, Tensor v, float mass) -> Tensor");

    m.def("dwc_dir_mxtsg_false(Tensor U, Tensor v, float mass, float csw) -> Tensor");

    m.def("dwc_fpre_mntsg_xtsghmn(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");
    m.def("dwc_fpre_mntsg_xtmghsn(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");
    m.def("dwc_fpre_mntsg_xtmnghs(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");
    m.def("dwc_fpre_mntsg_xtmdnghs(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");

    m.def("dw_block_mxtsg_dbxtsghm(Tensor U, Tensor v, float mass, int bls) -> Tensor");
    m.def("dw_block_mxtsg_bxtsghm(Tensor U, Tensor v, float mass, int bls) -> Tensor");

    m.def("dw_hop_mtsg_tMmgsh(Tensor U_ten, Tensor v_ten, Tensor hops_ten, float mass) -> Tensor");
    m.def("dw_hop_mtsg_tMgshm(Tensor U_ten, Tensor v_ten, Tensor hops_ten, float mass) -> Tensor");
    m.def("dw_hop_mtsg_tmgsMh(Tensor U_ten, Tensor v_ten, Tensor hops_ten, float mass) -> Tensor");
    m.def("dw_hop_mtsg_tmsgMh(Tensor U_ten, Tensor v_ten, Tensor hops_ten, float mass) -> Tensor");

    m.def("dw_hop_tmgs_tMmghs(Tensor U_ten, Tensor v_ten, Tensor hops_ten, float mass) -> Tensor");
    m.def("dw_hop_tmgs_tMmgsh(Tensor U_ten, Tensor v_ten, Tensor hops_ten, float mass) -> Tensor");
    m.def("dw_hop_tmgs_tMmgshu(Tensor U_ten, Tensor v_ten, Tensor hops_ten, float mass) -> Tensor");
    m.def("dw_hop_tmgs_tmgsMh(Tensor U_ten, Tensor v_ten, Tensor hops_ten, float mass) -> Tensor");

    m.def("dw_eo_pmtsg_pxtMmghs(Tensor Ue, Tensor Uo, Tensor ve, Tensor vo, float mass, int[] eodim) -> Tensor");

    m.def("dw_avx_tmgs_tmgsMhs(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dw_avx_mtsg_tmgsMhs(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");

    m.def("dwc_avx_tmgs_tmgsMhns(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass, float csw) -> Tensor");
    m.def("dwc_avx_mtsg_tmgsMhns(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass, float csw) -> Tensor");

    m.def("dw_templ_mtsg_tmgsMhs(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    
    m.def("dwc_templ_mtsg_tmgsMhns(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass, float csw) -> Tensor");

    m.def("dw_roof_templ_mtsg_tmgsMhs(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");

    m.def("dw_tempdir_mtsg_tmgsMhs(Tensor U, Tensor v, float mass) -> Tensor");

    m.def("dw_tempipe_mtsg_tmgsMhs(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");

    m.def("dw_templ_mtsgt_tmgsMht(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dwc_templ_mtsgt_tmngsMht(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass, float csw) -> Tensor");
    m.def("dwc_grid_mtsg_tmngsMhs(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dw_grid_mtsgt2_tmgsMht(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dwc_grid_mtsgt2_tmngsMht(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass) -> Tensor");

    m.def("dw_templ_mtsg_tmgsMhs_backw(Tensor U_tensor, Tensor grad_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dwc_grid_mtsg_backw(Tensor U_tensor, Tensor grad_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass) -> Tensor");

    m.def("dw_templ_mtsg_tmsgMhs(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dw_templbound_mtsg_tmsgMhs(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, Tensor bound_tensor, float mass) -> Tensor");
}

// Registers backend implementations
TORCH_LIBRARY_IMPL(qmad_history, CPU, m) {
    m.impl("dw_dir_mxtsg_xtsgMhm", &dw_dir_mxtsg_xtsgMhm);
    m.impl("dw_dir_mxtsg_xtMmghs", &dw_dir_mxtsg_xtMmghs);
    m.impl("dw_dir_mxtsg_xtMmdghs", &dw_dir_mxtsg_xtMmdghs);

    m.impl("dwc_dir_mxtsg_false", &dwc_dir_mxtsg_false);

    m.impl("dwc_fpre_mntsg_xtsghmn", &dwc_fpre_mntsg_xtsghmn);
    m.impl("dwc_fpre_mntsg_xtmghsn", &dwc_fpre_mntsg_xtmghsn);
    m.impl("dwc_fpre_mntsg_xtmnghs", &dwc_fpre_mntsg_xtmnghs);
    m.impl("dwc_fpre_mntsg_xtmdnghs", &dwc_fpre_mntsg_xtmdnghs);

    m.impl("dw_block_mxtsg_dbxtsghm", &dw_block_mxtsg_dbxtsghm);
    m.impl("dw_block_mxtsg_bxtsghm", &dw_block_mxtsg_bxtsghm);

    m.impl("dw_hop_mtsg_tMmgsh", &dw_hop_mtsg_tMmgsh);
    m.impl("dw_hop_mtsg_tMgshm", &dw_hop_mtsg_tMgshm);
    m.impl("dw_hop_mtsg_tmgsMh", &dw_hop_mtsg_tmgsMh);
    m.impl("dw_hop_mtsg_tmsgMh", &dw_hop_mtsg_tmsgMh_cpu);

    m.impl("dw_hop_tmgs_tMmghs", &dw_hop_tmgs_tMmghs);
    m.impl("dw_hop_tmgs_tMmgsh", &dw_hop_tmgs_tMmgsh);
    m.impl("dw_hop_tmgs_tMmgshu", &dw_hop_tmgs_tMmgshu);
    m.impl("dw_hop_tmgs_tmgsMh", &dw_hop_tmgs_tmgsMh);

    m.impl("dw_eo_pmtsg_pxtMmghs", &dw_eo_pmtsg_pxtMmghs);

    m.impl("dw_avx_tmgs_tmgsMhs", &dw_avx_tmgs_tmgsMhs);
    m.impl("dw_avx_mtsg_tmgsMhs", &dw_avx_mtsg_tmgsMhs);

    m.impl("dwc_avx_tmgs_tmgsMhns", &dwc_avx_tmgs_tmgsMhns);
    m.impl("dwc_avx_mtsg_tmgsMhns", &dwc_avx_mtsg_tmgsMhns);

    m.impl("dw_templ_mtsg_tmgsMhs", &dw_templ_mtsg_tmgsMhs);

    m.impl("dwc_templ_mtsg_tmgsMhns", &dwc_templ_mtsg_tmgsMhns);

    m.impl("dw_roof_templ_mtsg_tmgsMhs", &dw_roof_templ_mtsg_tmgsMhs);

    m.impl("dw_tempdir_mtsg_tmgsMhs", &dw_tempdir_mtsg_tmgsMhs);

    m.impl("dw_tempipe_mtsg_tmgsMhs", &dw_tempipe_mtsg_tmgsMhs);

    m.impl("dw_templ_mtsgt_tmgsMht", &dw_templ_mtsgt_tmgsMht);
    m.impl("dwc_templ_mtsgt_tmngsMht", &dwc_templ_mtsgt_tmngsMht);
    m.impl("dwc_grid_mtsg_tmngsMhs", &dwc_grid_mtsg_tmngsMhs);
    m.impl("dw_grid_mtsgt2_tmgsMht", &dw_grid_mtsgt2_tmgsMht);
    m.impl("dwc_grid_mtsgt2_tmngsMht", &dwc_grid_mtsgt2_tmngsMht);

    m.impl("dw_templ_mtsg_tmgsMhs_backw", &dw_templ_mtsg_tmgsMhs_backw);
    m.impl("dwc_grid_mtsg_backw", &dwc_grid_mtsg_backw);

    m.impl("dw_templ_mtsg_tmsgMhs", &dw_templ_mtsg_tmsgMhs);
    m.impl("dw_templbound_mtsg_tmsgMhs", &dw_templbound_mtsg_tmsgMhs);
}

}
