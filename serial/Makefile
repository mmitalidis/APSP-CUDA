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
all: oct_apsp_serial

oct_apsp_serial: oct_apsp_serial.o apsp_serial.o apsp_misc.o
	mkoctfile -I. -fPIC -o oct_apsp_serial apsp_misc.o apsp_serial.o oct_apsp_serial.o

oct_apsp_serial.o: oct_apsp_serial.cpp 
	g++ -c -fPIC -I$(OCTAVE_INCLUDE) -I$(OCTAVE_INCLUDE)octave -o oct_apsp_serial.o oct_apsp_serial.cpp 

apsp_serial.o: apsp_serial.c
	g++ -c -fPIC apsp_serial.c -o apsp_serial.o

apsp_misc.o: apsp_misc.c
	g++ -c -fPIC apsp_misc.c -o apsp_misc.o

clean:
	rm -f ./*.o

cleanAll:
	rm -f oct_apsp_serial.oct ./*.o 
