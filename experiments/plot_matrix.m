% Plot a boolean matrix in black-white.

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

% code from stackoverflow.com
function plot_matrix(mat)

	[r,c] = size(mat);
	figure('Position',[0,0, 1200,1200]);
	imagesc((1:c)+0.5,(1:r)+0.5,mat);
	colormap(gray); 
	axis equal 
	set(gca,'XTick',1:(c+1),'YTick',1:(r+1),... 
       	 	'XLim',[1 c+1],'YLim',[1 r+1],...
       	        'GridLineStyle','-','XGrid','on','YGrid','on');

end
