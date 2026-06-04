#include "cub_module_radix_sort.h"
#include <cub/cub.cuh>

#define CUDA_CHECK(call) \
{ \
  cudaError_t ierr = call; \
  if (cudaSuccess != ierr) \
  { \
    std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << std::endl \
              << "ERROR: " << cudaGetErrorString(ierr) << std::endl; \
    abort(); \
  } \
}

/// create a sequence of id0 + (0 to nvals - 1)
template <int nrnd>
__global__
void cub_module_radix_sort_initialize(int32_t *ids, int32_t id0, int32_t nvals)
{
  // TODO: can use int4 if nvals is divisible by 4
  #pragma unroll nrnd
  for (int32_t  q = blockIdx.x*blockDim.x + threadIdx.x;
       q < nvals; q += gridDim.x*blockDim.x)
  {
    ids[q] = q + id0;
  }
}

/// working space for device scan
void *g_cub_module_radix_sort_workspace = nullptr;
size_t g_cub_module_radix_sort_work_size = 0;
int g_cub_module_radix_sort_verbose = 0;

// --------------------------------------------------------------------------
void cub_module_radix_sort_allocate_workspace(int32_t num_elem)
{
  // check if we have enough work space
  size_t work_size = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, work_size,
                                             (int64_t*)nullptr, (int64_t*)nullptr,
                                             (int32_t*)nullptr, (int32_t*)nullptr,
                                             num_elem))
  if (g_cub_module_radix_sort_work_size < work_size)
  {
    // free old work space
    if (g_cub_module_radix_sort_workspace)
    {
      cudaFree(g_cub_module_radix_sort_workspace);
      g_cub_module_radix_sort_workspace = nullptr;
      g_cub_module_radix_sort_work_size = 0;
    }

    // allocate new work space
    g_cub_module_radix_sort_work_size = work_size;
    CUDA_CHECK(cudaMalloc(&g_cub_module_radix_sort_workspace,
                          g_cub_module_radix_sort_work_size))
    if (0 < g_cub_module_radix_sort_verbose)
      std::cerr << "allocated " << work_size << " bytes on the device" << std::endl;
  }
}

/// input an array of keys to sort
void cub_module_radix_sort(int64_t *keys_in, int64_t *keys_out,
                           int32_t *idx_in, int32_t *idx_out,
                           int32_t idx0, int32_t num_elem,
                           int32_t start_bit, int32_t end_bit)
{
  if (1 < g_cub_module_radix_sort_verbose)
    std::cerr << "cub_module_radix_sort" << std::endl;

  // allocate workspace
  cub_module_radix_sort_allocate_workspace(num_elem);

  // initialize an array with 1,2, .. N. after sort this array describes
  // the sorted order
  constexpr int nrnd = 4;
  constexpr int nthr = 128;
  int nblk = ( ((num_elem + nrnd - 1) / nrnd) + nthr - 1 ) / nthr;
  cub_module_radix_sort_initialize<nrnd><<<nblk,nthr>>>(idx_in, idx0, num_elem);

  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(g_cub_module_radix_sort_workspace,
                                             g_cub_module_radix_sort_work_size,
                                             keys_in, keys_out, idx_in, idx_out,
                                             num_elem, start_bit, end_bit))
}
