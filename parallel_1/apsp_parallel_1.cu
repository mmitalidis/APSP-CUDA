/*
 * =======================================================================
 *  This file is part of APSP-CUDA.
 *  Copyright (C) 2016 Marios Mitalidis
 *
 *  APSP-CUDA is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  APSP-CUDA is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with APSP-CUDA.  If not, see <http://www.gnu.org/licenses/>.
 * =======================================================================
 */ 


#include "apsp_parallel_1.h"
#include "cuda_error_check.h"

/*
 * Max matrix size is N = 2^12 = 4096. 
 * Total number of threads that will be executed is 4096^2.
 * One for each cell.
 */

// programmer defined
const int smp_executions = 8192;
const int threads_per_block = 128;
const int threads_per_smp = 2048;

// derived 
const int blocks_per_smp = threads_per_smp / threads_per_block;
const dim3 blocks(smp_executions, blocks_per_smp);
const dim3 threads(threads_per_block);

__global__ void apsp_parallel_1_kernel(float* dev_dist, int N, int k) {

	int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int i, j;
	float dist1, dist2, dist3;

	if (tid < N*N) {

		i = tid / N;
		j = tid - i*N;

		dist1 = dev_dist[tid];
		dist2 = dev_dist[i*N +k];
		dist3 = dev_dist[k*N +j];
	
		if (dist1 > dist2 + dist3)
			dev_dist[tid] = dist2 + dist3;
	}
}
 
// Solves the all-pairs shortest path problem using Floyd Warshall algorithm
void apsp_parallel_1(float** graph, float** dist, int N) {

	// allocate memory on the device
	float* dev_dist;
	gpuErrchk( cudaMalloc( (void**)&dev_dist, N*N * sizeof (float) ) );

	// initialize dist matrix on device
	for (int i = 0; i < N; i++)
		gpuErrchk( cudaMemcpy(dev_dist +i*N, graph[i], N * sizeof (float),
							  cudaMemcpyHostToDevice) );

	// For each element of the vertex set
	for (int k = 0; k < N; k++) {

		// launch kernel
		apsp_parallel_1_kernel<<<blocks,threads>>>(dev_dist,N,k);
		gpuKerchk();
    	}

	// return results to dist matrix on host
	for (int i = 0; i < N; i++)
		 gpuErrchk( cudaMemcpy(dist[i], dev_dist +i*N, N * sizeof (float),
							  cudaMemcpyDeviceToHost) );
}
 
