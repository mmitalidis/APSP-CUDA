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

% sections of code to execute
all = 1;
parallel = 1;
prob = 1;


% code, n, p, trial, time
fin = "apsp3_experiments.csv";
results = csvread(fin);

p = sort(unique(results(:,3)));
n = sort(unique(results(:,2)));
apps = sort(unique(results(:,1)));

tot_time = [];
for j = 1:length(apps)

	tot_time_cur = zeros(length(n),1);
	for i = 1:length(n)
		tot_time_cur(i) = mean(results(results(:,1) == apps(j) &
                                               results(:,2) == n(i)   , 5));
	end
	tot_time = [tot_time, tot_time_cur];
end

line = {'b-x', 'r-o', 'k-*', 'g-s'};

% all comparison
% ======================================================= %
if all
	figure(1);
	for i = 1:length(apps)
		semilogy(n,tot_time(:,i),line{i},"markersize",8,...
                                         "linewidth", 1); hold on;
	end
	yt = get(gca,"ytick");
	set(gca, 'yticklabel',sprintf('%d|',yt));

	grid minor;
	xlabel('Number of vertices (n)');
	ylabel('Elapsed time (s)');
	title('All-Pairs Shortest Path - Serial / GPU Parallel Comparison');

	% add legend
	if length(apps) == 4
		legend({'Serial','One cell, no shared memory.',...
		        'One cell, with shared memory.',...
	        	'Multiple cells, with shared memory.'},...
                	'location','northwest');
	end
  
	print -djpg all_comparison.jpg;
end

% parallel comparison
% ======================================================= %
if parallel
	if length(apps) == 4
		figure(2);
		for i = 2:length(apps)
			semilogy(n,tot_time(:,i),line{i},"markersize",8,...
       	                                         	"linewidth", 1); hold on;
		end
		yt = get(gca,"ytick");
		set(gca, 'yticklabel',sprintf('%d|',yt));
	
		grid minor;
		xlabel('Number of vertices (n)');
		ylabel('Elapsed time (s)');
		title('All-Pairs Shortest Path - GPU Parallel Comparison');
	
		legend({'One cell, no shared memory.',...
       	        	'One cell, with shared memory.',...
        		'Multiple cells, with shared memory.'},...
       	 		'location','northwest');
 	 
		print -djpg parallel_comparison.jpg;
	end
end

% edge probability comparison
% ======================================================= %
if prob
	if length(apps) == 4

		rel_time_0 = [];
		rel_time_1 = [];
		rel_time_2 = [];
		rel_time_3 = [];

		for i = 1:length(p)
			rel_time_cur_0 = zeros(length(n),1);
			rel_time_cur_1 = zeros(length(n),1);
			rel_time_cur_2 = zeros(length(n),1);
			rel_time_cur_3 = zeros(length(n),1);

			for j = 1:length(n)
			rel_time_cur_0(j) = results( results(:,1) == 0  &
						   results(:,2) == n(j) &
						   results(:,3) == p(i),5);

			rel_time_cur_1(j) = results( results(:,1) == 1  &
						   results(:,2) == n(j) &
						   results(:,3) == p(i),5);

			rel_time_cur_2(j) = results( results(:,1) == 2  &
						   results(:,2) == n(j) &
						   results(:,3) == p(i),5);

			rel_time_cur_3(j) = results( results(:,1) == 3  &
						   results(:,2) == n(j) &
						   results(:,3) == p(i),5);

			end
			rel_time_0 = [rel_time_0, rel_time_cur_0];
			rel_time_1 = [rel_time_1, rel_time_cur_1];
			rel_time_2 = [rel_time_2, rel_time_cur_2];
			rel_time_3 = [rel_time_3, rel_time_cur_3];
		end

		for i = 2:length(p)
		rel_time_0(:,i) = (rel_time_0(:,i) - rel_time_0(:,1)) ./...
				   rel_time_0(:,1) * 100;
		rel_time_1(:,i) = (rel_time_1(:,i) - rel_time_1(:,1)) ./...
				   rel_time_1(:,1) * 100;
		rel_time_2(:,i) = (rel_time_2(:,i) - rel_time_2(:,1)) ./...
				   rel_time_2(:,1) * 100;
		rel_time_3(:,i) = (rel_time_3(:,i) - rel_time_3(:,1)) ./...
					   rel_time_3(:,1) * 100;
		end

		rel_time_0(:,1) = zeros(length(n),1);
		rel_time_1(:,1) = zeros(length(n),1);
		rel_time_2(:,1) = zeros(length(n),1);
		rel_time_3(:,1) = zeros(length(n),1);

		legend_text = {};
		for i = 1:length(p)
			legend_text{i} = ["p = " num2str(p(i))];
		end
		line = {'b-x', 'r-o', 'k-*'};


		% ---------------------------------------------

		n = n(3:end);
		rel_time_0 = rel_time_0(3:end,:);
		rel_time_1 = rel_time_1(3:end,:);
		rel_time_2 = rel_time_2(3:end,:);
		rel_time_3 = rel_time_3(3:end,:);

		subplot(2,2,1);
		for i = 1:length(p)
			plot(n,rel_time_0(:,i),line{i},"markersize",8,...
					       "linewidth",1); hold on;
		end
		yt = get(gca,"ytick");
		set(gca, 'yticklabel',sprintf('%d|',yt));

		title('Serial relative time comparison');
		xlabel('Number of vertices (n)');
		ylabel('Percent of relative time difference');
		grid on;
		%legend(legend_text,'location','northwest');

		% ---------------------------------------------

		subplot(2,2,2);
		for i = 1:length(p)
			plot(n,rel_time_1(:,i),line{i},"markersize",8,...
					       "linewidth",1); hold on;
		end
		yt = get(gca,"ytick");
		set(gca, 'yticklabel',sprintf('%d|',yt));

		title('Parallel (one cell, no shared mem) rel. time');
		xlabel('Number of vertices (n)');
		ylabel('Percent of relative time difference');
		grid on;
		legend(legend_text,"location","southeast");

		% ---------------------------------------------

		subplot(2,2,3);
		for i = 1:length(p)
			plot(n,rel_time_2(:,i),line{i},"markersize",8,...
					       "linewidth",1); hold on;
		end
		yt = get(gca,"ytick");
		set(gca, 'yticklabel',sprintf('%d|',yt));

		title('Parallel (multiple cells, no shared mem) rel. time');
		xlabel('Number of vertices (n)');
		ylabel('Percent of relative time difference');
		grid on;
		%legend(legend_text,"location","northwest");

		% ---------------------------------------------

		subplot(2,2,4);
		for i = 1:length(p)
			plot(n,rel_time_3(:,i),line{i},"markersize",8,...
					       "linewidth",1); hold on;
		end
		yt = get(gca,"ytick");
		set(gca, 'yticklabel',sprintf('%d|',yt));

		title('Parallel (multiple cells, no shared mem) rel. time');
		xlabel('Number of vertices (n)');
		ylabel('Percent of relative time difference');
		grid on;
		%legend(legend_text,"location","northwest");

		print -djpg rel_time_comparison.jpg;
	end

end
