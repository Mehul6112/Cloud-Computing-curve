#include <iostream>
#include <queue>
#include <omp.h>

#define MAX_NODES 1000

int adj[MAX_NODES][MAX_NODES]; // adjacency matrix
int num_nodes;

void add_edge(int u, int v) {
    adj[u][v] = 1;
    adj[v][u] = 1; // undirected graph
}

void parallel_bfs(int start) {
    bool visited[MAX_NODES] = {false};
    int current_frontier[MAX_NODES];
    int next_frontier[MAX_NODES];
    int current_size = 0, next_size = 0;

    visited[start] = true;
    current_frontier[current_size++] = start;

    while (current_size > 0) {
        next_size = 0;

        #pragma omp parallel for
        for (int i = 0; i < current_size; ++i) {
            int node = current_frontier[i];

            #pragma omp critical
            std::cout << "Visited: " << node << "\n";

            for (int neighbor = 0; neighbor < num_nodes; ++neighbor) {
                if (adj[node][neighbor] && !visited[neighbor]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            next_frontier[next_size++] = neighbor;
                        }
                    }
                }
            }
        }

        // Prepare for next level
        for (int i = 0; i < next_size; ++i) {
            current_frontier[i] = next_frontier[i];
        }
        current_size = next_size;
    }
}

int main() {
    num_nodes = 6;

    // Example graph
    add_edge(0, 1);
    add_edge(0, 2);
    add_edge(1, 3);
    add_edge(1, 4);
    add_edge(2, 4);
    add_edge(3, 5);
    add_edge(4, 5);

    std::cout << "Starting Parallel BFS from node 0:\n";
    parallel_bfs(0);

    return 0;
}
