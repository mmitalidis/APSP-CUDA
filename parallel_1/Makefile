# =======================================================================
#  This file is part of APSP-CUDA.
#  Copyright (C) 2016 Marios Mitalidis
#
#  APSP-CUDA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  APSP-CUDA is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with APSP-CUDA.  If not, see <http://www.gnu.org/licenses/>.
# =======================================================================

OCTAVE_INCLUDE = /usr/include/octave-4.0.0/
.DEFAULT_GOAL = all

makefile:
all: oct_apsp_parallel_1

oct_apsp_parallel_1: oct_apsp_parallel_1.o apsp_parallel_1.o apsp_misc.o cuda_error_check.o
	mkoctfile -I. -fPIC -lcudart -o oct_apsp_parallel_1 cuda_error_check.o apsp_parallel_1.o apsp_misc.o oct_apsp_parallel_1.o

oct_apsp_parallel_1.o: oct_apsp_parallel_1.cpp
	g++ -c -fPIC -I$(OCTAVE_INCLUDE) -I$(OCTAVE_INCLUDE)octave -o oct_apsp_parallel_1.o oct_apsp_parallel_1.cpp

apsp_parallel_1.o: apsp_parallel_1.cu 
	nvcc -c -Xcompiler -fPIC apsp_parallel_1.cu -o apsp_parallel_1.o

apsp_misc.o: apsp_misc.cu
	nvcc -c -Xcompiler -fPIC apsp_misc.cu -o apsp_misc.o

cuda_error_check.o: cuda_error_check.cu
	nvcc -c -Xcompiler -fPIC cuda_error_check.cu -o cuda_error_check.o

clean:
	rm -f ./*.o

cleanAll:
	rm -f oct_apsp_parallel_1.oct ./*.o 
