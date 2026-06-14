#include "cub_inclusive_scan.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>


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

/// working space for device scan
void *g_workspace = nullptr;
size_t g_work_size = 0;
int g_verbose = 0;

// --------------------------------------------------------------------------
void cub_allocate_workspace(int num_octs)
{
  // check if we have enough work space
  size_t work_size = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, work_size,
                                           (int*)nullptr, (int*)nullptr, num_octs))
  if (g_work_size < work_size)
  {
    // free old work space
    if (g_workspace)
    {
      cudaFree(g_workspace);
      g_workspace = nullptr;
      g_work_size = 0;
    }

    // allocate new work space
    g_work_size = work_size;
    CUDA_CHECK(cudaMalloc(&g_workspace, g_work_size))
    if (0 < g_verbose)
      std::cout << "allocated " << work_size << " bytes on the device" << std::endl;
  }
}

// --------------------------------------------------------------------------
void cub_inclusive_sum(int *prefix_sum, int head_idx, int num_octs)
{
  if (1 < g_verbose)
    std::cerr << "cub_inclusive_sum" << std::endl;

  // index into the array as done in RAMSES
  int *prefix_sum_loc = prefix_sum + head_idx - 1;

  // check if we have enough work space, and allocate if needed
  cub_allocate_workspace(num_octs);

  // compute the prefix scan
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(g_workspace, g_work_size,
                                           prefix_sum_loc, prefix_sum_loc, num_octs))
}
