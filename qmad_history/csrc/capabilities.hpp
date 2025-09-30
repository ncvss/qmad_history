#include <torch/extension.h>
#include <vector>

namespace qmad_history {

// return a tensor whose entries denote if these macros are defined
std::vector<int64_t> capability_function (const at::Tensor& dummy){

    int64_t vectorisation = 0;
    int64_t parallelisation = 0;
    int64_t cuda_error = 0;

#ifdef VECTORISATION_ACTIVATED
    vectorisation = 1;
#endif

#ifdef PARALLELISATION_ACTIVATED
    parallelisation = 1;
#endif

#ifdef ERROR_HANDLING_OUTPUT
    cuda_error = 1;
#endif

    std::vector<int64_t> out = {vectorisation, parallelisation, cuda_error};

    return out;
}

}

