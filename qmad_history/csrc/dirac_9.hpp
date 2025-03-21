#include <torch/extension.h>

namespace qmad_history{

/**
 * @brief Dirac Wilson operator using vectorisation with the Grid memory layout
 *        (2 neighbouring sites in t direction are in one register,
 *        this is the fastest runnning index in memory)
 * 
 * @param U_tensor gauge field
 * @param v_tensor vector field
 * @param hops_tensor addresses for the gauge hops
 * @param mass mass parameter
 * @return at::Tensor 
 */
at::Tensor dw_templ_mtsgt_tmgsMht (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                   const at::Tensor& hops_tensor, double mass);

/**
 * @brief Dirac Wilson Clover operator using vectorisation with the Grid memory layout
 *        (2 neighbouring sites in t direction are in one register,
 *        this is the fastest runnning index in memory)
 * 
 * @param U_tensor gauge field
 * @param v_tensor vector field
 * @param fs_tensors field strength tensor product with sigma,
 *                   only the upper triangles of two 6x6 matrices are passed
 * @param hops_tensor addresses for the gauge hops
 * @param mass mass parameter
 * @param csw Sheikholeslami-Wohlert coefficient
 * @return at::Tensor 
 */
at::Tensor dwc_templ_mtsgt_tmngsMht (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                    const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                                    double mass, double csw);

}
