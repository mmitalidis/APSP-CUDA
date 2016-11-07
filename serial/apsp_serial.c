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


#include "apsp_serial.h"

// Solves the all-pairs shortest path problem using Floyd Warshall algorithm
// Part of code from:
// http://www.geeksforgeeks.org/dynamic-programming-set-16-floyd-warshall-algorithm/
void apsp_serial (float** graph, float** dist, int N) {

	// dist[][] will be the output matrix that will finally have the shortest 
   	// distances between every pair of vertices
	int i, j, k;

	// initialize dist matrix
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			dist[i][j] = graph[i][j];
 
	// For each element of the vertex set
	for (k = 0; k < N; k++) {
        
		// Pick all vertices as source one by one
        	for (i = 0; i < N; i++) {

			// Pick all vertices as destination for the above picked source
            		for (j = 0; j < N; j++) {
                
				// If vertex k is on the shortest path from
                		// i to j, then update the value of dist[i][j]
                		if (dist[i][k] + dist[k][j] < dist[i][j])
                    			dist[i][j] = dist[i][k] + dist[k][j];
            		}
        	}
    	}
}
 
