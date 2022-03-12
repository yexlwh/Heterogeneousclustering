function [F] = FloydWarshall(D)
% Solve the all-pairs shortest path problem using Floyd Warshall algorithm
% F will be the output matrix that will finally have the shortest distances between every pair of vertices
   
V = size(D, 1);

% Initialize the solution matrix same as input graph matrix. Or we can say the initial values of shortest distances are based on shortest paths considering no intermediate vertex. 
F = D;

% Add all vertices one by one to the set of intermediate vertices. Before start of a iteration, we have shortest distances between all pairs of vertices such that the shortest distances consider only the vertices in set {0, 1, 2, ... k-1} as intermediate vertices. After the end of a iteration, vertex no. k is added to the set of intermediate vertices and the set becomes {0, 1, 2, ... k}.     

for k=1:V
    % Pick all vertices as source one by one
    for i=1:V
        % Pick all vertices as destination for the above picked source
        for j=1:V
            % If vertex k is on the shortest path from i to j, then update the value of dist[i][j]
            if ( F(i,k) +  F(k,j) ) <  F(i,j) 
               F(i,j) = F(i,k) + F(k,j);
            end
        end
    end
end 