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


// Code from:
// http://www.geeksforgeeks.org/dynamic-programming-set-16-floyd-warshall-algorithm/

#include <stdio.h>
#include "apsp_serial.h"
 
// Number of vertices in the graph
#define V 4
#define INF 99999
 
/* A utility function to print solution */
void printSolution(float** dist, int N) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (dist[i][j] == INF)
                printf("%7s", "INF");
            else
                printf ("%7.1f", dist[i][j]);
        }
        printf("\n");
    }
}
 
// driver program to test above function
int main()
{
    /* Let us create the following weighted graph
            10
       (0)------->(3)
        |         /|\
      5 |          |
        |          | 1
       \|/         |
       (1)------->(2)
            3           */
    	float** g;
	g= matrix_malloc(V);	

	// row 0
     	g[0][0] = 0;  
     	g[0][1] = 5;  
     	g[0][2] = INF;  
     	g[0][3] = 10;  

	// row 1
     	g[1][0] = INF;
     	g[1][1] = 0;  
     	g[1][2] = 3;  
     	g[1][3] = INF;  

	// row 2
     	g[2][0] = INF;  
     	g[2][1] = INF;  
     	g[2][2] = 0;  
     	g[2][3] = 1;  

	// row 3
     	g[3][0] = INF;  
     	g[3][1] = INF;  
     	g[3][2] = INF;  
     	g[3][3] = 0;  

	// allocate memory for dist matrix
	float** d = matrix_malloc(V);

	// execute algorithm
	apsp_serial(g,d,V);

    	// Print the solution
    	printSolution(d,V);
    	return 0;
}
