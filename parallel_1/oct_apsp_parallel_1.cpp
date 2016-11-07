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


#include <octave/oct.h>
#include "dMatrix.h"
	
#include "apsp_parallel_1.h"
#include "apsp_misc.h"

#include <time.h>
#include <sys/time.h>

DEFUN_DLD (oct_apsp_parallel_1, args, nargout,
           "All Pair Shortest Path (parallel 1)") {

	char usage[] = "Provide one NxN matrix.\n";
	octave_value_list retval;

	// check number of arguments
	int nargin = args.length ();
	if (nargin != 1) {
		error(usage);
		return retval;
	}

	// get input matrix
	Matrix gm( args(0).matrix_value() );
	if (error_state) {
		error(usage);
		return retval;
	}
	
	// get dimensions of input matrix
	int N = gm.rows();
	int M = gm.columns();

	if (N != M) {
		error(usage);
		return retval;
	}

	// for time measurements
	struct timeval startwtime, endwtime;
	double tot_time; 

	float** g = matrix_malloc(N);
	float** d = matrix_malloc(N);

	// initialize g matrix
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			g[i][j] = (float) gm(i,j);
		}
	}

	// start measuring time
	gettimeofday(&startwtime,NULL);

	// execute serial apsp
	apsp_parallel_1(g,d,N);

	// stop measuring time
	gettimeofday(&endwtime,NULL);

	// calculate time
	tot_time = (double) ( (endwtime.tv_usec - startwtime.tv_usec) / 1.0e6
              	+ endwtime.tv_sec - startwtime.tv_sec );


	// initialize output matrix
	Matrix dm(gm);

	// store dm matrix
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			dm(i,j) = (double) d[i][j];
		}
	}
		
	// free memory for intermediate calculation
	matrix_free(g,N);
	matrix_free(d,N);

	retval(0) = octave_value(dm);
	retval(1) = octave_value(tot_time);

	return retval; 
}
