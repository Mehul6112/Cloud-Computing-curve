#include <iostream>
#include <vector>
#include <limits>
#include <omp.h>

using ll = long long;
const ll INF = std::numeric_limits<ll>::max();

// Graph represented as adjacency list: graph[u] = vector of (v, weight)
using Graph = std::vector<std::vector<std::pair<int,int>>>;

std::vector<ll> dijkstra(const Graph& graph, int source) {
    int V = graph.size();
    std::vector<ll> dist(V, INF);
    std::vector<bool> visited(V, false);

    dist[source] = 0;

    for (int iter = 0; iter < V; ++iter) {
        // 1) Find unvisited vertex u with smallest dist[u]
        int u = -1;
        ll best = INF;
        for (int i = 0; i < V; ++i) {
            if (!visited[i] && dist[i] < best) {
                best = dist[i];
                u = i;
            }
        }
        if (u == -1) break;           // remaining vertices unreachable
        visited[u] = true;

        // 2) Relax all edges out of u in parallel
        auto& nbrs = graph[u];
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < (int)nbrs.size(); ++k) {
            int v = nbrs[k].first;
            int w = nbrs[k].second;
            if (!visited[v]) {
                ll nd = best + w;
                // only one thread writes to dist[v] at a time,
                // and each v appears only once in this loop
                if (nd < dist[v]) {
                    dist[v] = nd;
                }
            }
        }
    }

    return dist;
}

int main() {
    // Example graph (6 nodes) from your adjacency‑matrix example:
    // 0—4—1—8—2—7—3—9—4—10—5—4—2 and 3—14—5
    Graph graph(6);
    auto add_edge = [&](int u, int v, int w){
        graph[u].emplace_back(v,w);
        graph[v].emplace_back(u,w);
    };
    add_edge(0,1,4);
    add_edge(1,2,8);
    add_edge(2,3,7);
    add_edge(3,4,9);
    add_edge(4,5,10);
    add_edge(2,5,4);
    add_edge(3,5,14);

    int source = 0;
    double t0 = omp_get_wtime();
    auto dist = dijkstra(graph, source);
    double t1 = omp_get_wtime();

    std::cout << "Shortest distances from source node " << source << ":\n";
    for (int i = 0; i < (int)dist.size(); ++i) {
        std::cout << "Node " << i << ": ";
        if (dist[i] == INF) std::cout << "∞\n";
        else                std::cout << dist[i] << "\n";
    }
    std::cout << "Computed in " << (t1 - t0) << " seconds.\n";
    return 0;
}
