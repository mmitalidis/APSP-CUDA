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


#include "apsp_misc.h"

const int errorMemAllocate = 101;

//free matrix data
void  matrix_free(float** mat, int N) {
	for (int i = 0; i < N; i++)
		free(mat[i]);
	free(mat);
}

// safe data allocation
void* safe_malloc(size_t size) {
	void* ptr = malloc(size);
	if (ptr == NULL) {
		fprintf(stderr,"Cannot allocate memory\n");
		exit(errorMemAllocate);
	}

	return ptr;
}

// allocate memory for matrix
float** matrix_malloc(int N) {
	float** mat = (float**) safe_malloc(N * sizeof(float*));
	for (int i = 0; i < N; ++i)
		mat[i] = (float*) safe_malloc(N * sizeof(float)); 
	return mat;
}
