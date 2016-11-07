% make adjacency matrix A(1:n,1:n) where a edge is generated with probability p
% and random edge weights (0:w)
%
% e.g. A = makeAdjacency(10,0.7,30) makes a 10x10 adjacency matrix with
% edge weights 0:30 with 0.7 probability

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

function A = makeAdjacency(n,p,w)

A = zeros(n);

for i=1:n
  for j=1:n
    if rand()>p
      A(i,j) = inf;
    else
      A(i,j) = rand()*w;
    end
  end
  A(i,i) = 0;
end
