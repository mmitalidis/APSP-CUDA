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


#include "apsp_parallel_2.h"
#include "apsp_misc.h"
#include "cuda_error_check.h"


#define MIN_MACRO(a,b) ( (a) < (b) ? (a) : (b) )


/*
 * Max matrix size is N = 2^12 = 4096. 
 * Total number of threads that will be executed is 4096^2.
 * One for each cell.
 */

// programmer defined
const int block_dim = 32;
const dim3 threads(block_dim,block_dim);

// kernel for pass 1
__global__ void apsp_parallel_2_kernel_1(float* dev_dist, int N, int stage) {

	// get indices for the current cell
	int i = stage*block_dim + threadIdx.y;
	int j = stage*block_dim + threadIdx.x;

	int tid = i*N + j;

	// allocate shared memory
	__shared__ float sd[block_dim][block_dim];

	// copy data from main memory to shared memory
	sd[threadIdx.y][threadIdx.x] = dev_dist[tid];
	__syncthreads();

	// iterate for the values of k
	for (int k = 0; k < block_dim; k++) {

		float vertex   = sd[threadIdx.y][threadIdx.x];
		float alt_path = sd[k][threadIdx.x] + sd[threadIdx.y][k];

		sd[threadIdx.y][threadIdx.x] = MIN_MACRO( vertex, alt_path );
		__syncthreads();

	}

	// write result back to main memory
	dev_dist[tid] = sd[threadIdx.y][threadIdx.x];
}

// kernel for pass 2
__global__ void apsp_parallel_2_kernel_2(float* dev_dist, int N, int stage) {
	
	// get indices of the current block
	int skip_center_block = MIN_MACRO( (blockIdx.x+1)/(stage+1), 1 );

	int box_y = 0;
	int box_x = 0;

	// block in the same row with the primary block
	if (blockIdx.y == 0) {
		box_y = stage;
		box_x = blockIdx.x + skip_center_block;
	
	}
	// block in the same column with the primary block
	else {
		box_y = blockIdx.x + skip_center_block;
		box_x = stage;
	}

	// get indices for the current cell
	int i = box_y * block_dim + threadIdx.y;
	int j = box_x * block_dim + threadIdx.x;

	// get indices for the cell of the primary block
	int pi = stage*block_dim + threadIdx.y;
	int pj = stage*block_dim + threadIdx.x;

	// get indices of the cells from the device main memory
	int tid = i*N + j;
	int ptid = pi*N + pj;

	// allocate shared memory
	__shared__ float sd[block_dim][2*block_dim];

	// copy current block and primary block to shared memory
	sd[threadIdx.y][threadIdx.x]             = dev_dist[tid];
	sd[threadIdx.y][block_dim + threadIdx.x] = dev_dist[ptid]; 
	__syncthreads();

	// block in the same row with the primary block
	if (blockIdx.y == 0) {
		for (int k = 0; k < block_dim; k++) {

			float vertex   = sd[threadIdx.y][threadIdx.x];
			float alt_path = sd[k][threadIdx.x] 
                               	              + sd[threadIdx.y][block_dim + k];

			sd[threadIdx.y][threadIdx.x] = MIN_MACRO( vertex, alt_path );
			__syncthreads();

		}
	}
	// block in the same column with the primary block
	else {
		for (int k = 0; k < block_dim; k++) {

			float vertex   = sd[threadIdx.y][threadIdx.x];
			float alt_path = sd[threadIdx.y][k] 
                                           + sd[k][block_dim + threadIdx.x];

			sd[threadIdx.y][threadIdx.x] = MIN_MACRO( vertex, alt_path );
			__syncthreads();
		}
	}

	// write result back to main memory
	dev_dist[tid] = sd[threadIdx.y][threadIdx.x];
}

// kernel for pass 3
__global__ void apsp_parallel_2_kernel_3(float* dev_dist, int N, int stage) {

	// get indices of the current block
	int skip_center_block_y = MIN_MACRO( (blockIdx.y+1)/(stage+1), 1 );
	int skip_center_block_x = MIN_MACRO( (blockIdx.x+1)/(stage+1), 1 );

	int box_y = blockIdx.y + skip_center_block_y;
	int box_x = blockIdx.x + skip_center_block_x;

	// get indices for the current cell
	int i = box_y * block_dim + threadIdx.y;
	int j = box_x * block_dim + threadIdx.x;

	// get indices from the cell in the same row with the current box
	int ri = i;
	int rj = stage*block_dim + threadIdx.x;
	
	// get indices from the cell in the same column with the current box
	int ci = stage*block_dim + threadIdx.y;
	int cj = j;

	// get indices of the cells from the device main memory
	int  tid =  i*N +  j;
	int rtid = ri*N + rj;
	int ctid = ci*N + cj;

	// allocate shared memory
	__shared__ float sd[block_dim][3*block_dim];

	// copy current block and depending blocks to shared memory
	sd[threadIdx.y][threadIdx.x]               = dev_dist[tid];
	sd[threadIdx.y][  block_dim + threadIdx.x] = dev_dist[rtid]; 
	sd[threadIdx.y][2*block_dim + threadIdx.x] = dev_dist[ctid]; 
	__syncthreads();

	for (int k = 0; k < block_dim; k++) {

		float vertex   = sd[threadIdx.y][threadIdx.x];
		float alt_path = sd[threadIdx.y][block_dim + k]
       	        	             + sd[k][2*block_dim + threadIdx.x];

		sd[threadIdx.y][threadIdx.x] = MIN_MACRO( vertex, alt_path );
		__syncthreads();
	}

	// write result back to main memory
	dev_dist[tid] = sd[threadIdx.y][threadIdx.x];
}

/*
 * Solves the all-pairs shortest path problem using Floyd Warshall algorithm
 *
 * Implementation based on:
 * All-Pairs Shortest-Paths for Large Graphs on the GPU
 * Gary J. Katz and Joseph T. Kider Jr.
*/
int apsp_parallel_2(float** graph, float** dist, int N) {

	// check the dimension of the input matrix
	if (!isPowerOfTwo(N) || N < 128) {
		return (apsp_parallel_2_status::invalid_dimension);
	}

	// allocate memory on the device
	float* dev_dist;
	gpuErrchk( cudaMalloc( (void**)&dev_dist, N*N * sizeof (float) ) );

	// initialize dist matrix on device
	for (int i = 0; i < N; i++)
		gpuErrchk( cudaMemcpy(dev_dist +i*N, graph[i], N * sizeof (float),
							  cudaMemcpyHostToDevice) );
	// get the power of 2 of the dimension
	int p = getPowerofTwo(N);
	int r = getPowerofTwo(block_dim);

	int nBlocks = 1 << (p-r);
	
	// get the dimensions of the grid
	dim3 blocks1(1);
	dim3 blocks2(nBlocks-1,2);
	dim3 blocks3(nBlocks-1, nBlocks-1);

	// For each element of the vertex set
	for (int stage = 0; stage < nBlocks; stage++) {

		// pass 1 - launch kernel 1
		apsp_parallel_2_kernel_1<<<blocks1,threads>>>(dev_dist,N,stage);
		gpuKerchk();

		// pass 2 - launch kernel 2
		apsp_parallel_2_kernel_2<<<blocks2,threads>>>(dev_dist,N,stage);
		gpuKerchk();

		// pass 3 - launch kernel 3
		apsp_parallel_2_kernel_3<<<blocks3,threads>>>(dev_dist,N,stage);
		gpuKerchk();
 	}

	// return results to dist matrix on host
	for (int i = 0; i < N; i++)
		 gpuErrchk( cudaMemcpy(dist[i], dev_dist +i*N, N * sizeof (float),
							  cudaMemcpyDeviceToHost) );
	return (apsp_parallel_2_status::success);
}
 
