#!/usr/bin/octave-cli -qf

% =======================================================================
%  This file is part of APSP-CUDA.
%  Copyright (C) 2016 Marios Mitalidis
%
%  APSP-CUDA is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
%
%  APSP-CUDA is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with APSP-CUDA.  If not, see <http://www.gnu.org/licenses/>.
% =======================================================================

% file to write results
fout = "apsp3_experiments.csv";

% max edge weight
w = 1;

% probability of generating an edge
p = [0.33, 0.45, 0.66];

% number of vertices
q = 7:12;
n = 2.^q;

% number of trials
t = 3;

results = [];

% run tests
for i = 1:length(n)

	disp(["[*] Running Experiments for n = " num2str(n(i))]);
	for j = 1:length(p)
		for k = 1:length(t)

			A = makeAdjacency(n(i),p(j),w);
			
			[~,t0] = oct_apsp_serial(A);
			[~,t1] = oct_apsp_parallel_1(A);
			[~,t2] = oct_apsp_parallel_2(A);
			[~,t3] = oct_apsp_parallel_3(A);
			
			results = [results; 
				0, n(i), p(j), t, t0;
				1, n(i), p(j), t, t1;
				2, n(i), p(j), t, t2;
				3, n(i), p(j), t, t3];

		end
	end
end

csvwrite(fout, results);
