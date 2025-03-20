// in this wilson, I do what I suppose GPT does:
// the spatial loop goes over the input grid sites, makes all computations
// that use that input site, and writes them to the corresponding site
// in the output tensor
// however, I think this does not reduce the number of operations
// thus this gets abandoned

// #include <torch/extension.h>
// #include <omp.h>

// #include "static/indexfunc_2.hpp"
// #include "static/gamma_1.hpp"

// namespace qmad_history {

// at::Tensor dw_inloop_mtsg_tmgsMh (const at::Tensor& U_ten, const at::Tensor& v_ten,
//                                const at::Tensor& hops_ten, double mass){
    
//     TORCH_CHECK(v_ten.dim() == 6);
//     TORCH_CHECK(U_ten.size(1) == v_ten.size(0));
//     TORCH_CHECK(U_ten.size(2) == v_ten.size(1));
//     TORCH_CHECK(U_ten.size(3) == v_ten.size(2));
//     TORCH_CHECK(U_ten.size(4) == v_ten.size(3));
//     TORCH_CHECK(v_ten.size(4) == 4);
//     TORCH_CHECK(v_ten.size(5) == 3);
    
//     TORCH_CHECK(U_ten.is_contiguous());
//     TORCH_CHECK(v_ten.is_contiguous());
//     TORCH_CHECK(hops_ten.is_contiguous());

//     int vol = hops_ten.size(0);

//     // this time, zero out result, as we access the result sites multiple times
//     at::Tensor result_ten = torch::zeros(v_ten.sizes(), v_ten.options());
//     const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
//     const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
//     const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
//     c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

// #pragma omp parallel for
//     for (int t = 0; t < vol; t++){
        
//         // mass term and mu = 0 term in same loop to minimize result access
//         int mu = 0;
//         for (int g = 0; g < 3; g++){
//             for (int s = 0; s < 4; s++){

//                 // mass term stays the same, as it is the same site for in and out
//                 result[vixo(t,g,s)] += (4.0 + mass) * v[vixo(t,g,s)];

//                 for (int gi = 0; gi < 3; gi++){
                    
//                     // the term of 
//                     c10::complex<double> hopplus = v[vixo(t,g,s)]

//                     result[vixo(hops[hix(t,mu,0)],g,s)]
                    
//                     result[vixo(t,g,s)] += (
//                         std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
//                         * (
//                             -v[vixo(hops[hix(t,mu,0)],gi,s)]
//                             -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
//                         )
//                         + U[uixo(t,mu,g,gi,vol)]
//                         * (
//                             -v[vixo(hops[hix(t,mu,1)],gi,s)]
//                             +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
//                         )
//                     ) * 0.5;
//                 }
//             }
//         }

//         for (mu = 1; mu < 4; mu++){
//             for (int g = 0; g < 3; g++){
//                 for (int s = 0; s < 4; s++){
//                     for (int gi = 0; gi < 3; gi++){
                        
//                         result[vixo(t,g,s)] += (
//                             std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
//                             * (
//                                 -v[vixo(hops[hix(t,mu,0)],gi,s)]
//                                 -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
//                             )
//                             + U[uixo(t,mu,g,gi,vol)]
//                             * (
//                                 -v[vixo(hops[hix(t,mu,1)],gi,s)]
//                                 +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
//                             )
//                         ) * 0.5;
//                     }
//                 }
//             }
//         }
//     }

//     return result_ten;
// }

// }
