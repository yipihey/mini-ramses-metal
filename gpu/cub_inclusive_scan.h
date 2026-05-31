#ifndef inclusive_scan_h
#define inclusive_scan_h

extern "C"
{

/** pre-allocate work space for scan. note that the scan itself will always
 * check and allocate if needed.
 */
void cub_allocate_workspace(int num_octs);

/** compute the inclusive prefix sum.
 * @param[in] prefix_sum array of values to compute prefix sum on
 * @param[in] head_idx offset to begin scan at
 * @param[in] num_octs number of array elements to scan
 */
void cub_inclusive_sum(int *prefix_sum, int head_idx, int num_octs);

}

#endif
