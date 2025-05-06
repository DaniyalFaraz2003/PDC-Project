#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <limits>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

using namespace std;

struct Vertex {
    int id;
    vector<Vertex*> neighbors;
    float weight = 1.0f;
};

class Graph {
private:
    map<int, Vertex> adjList;
    int numVertices;
    int numEdges;

public:
    Graph() {
        adjList.clear();
        numVertices = 0;
        numEdges = 0;
    }

    int getNumVertices() { return numVertices; }

    void buildAdjacencyList(string fileName) {
        ifstream file(fileName);
        if (!file.is_open()) {
            cerr << "Error opening file: " << fileName << endl;
            return;
        }
    
        string line;
        while (getline(file, line)) {
            // Skip comment lines
            if (line.empty() || line[0] == '#') {
                continue;
            }
    
            int u, v;
            istringstream iss(line);
            if (!(iss >> u >> v)) {
                continue;
            }

            u++;
            v++;
    
            if (adjList.find(u) == adjList.end()) {
                adjList[u] = Vertex{u, {}};
            }
            if (adjList.find(v) == adjList.end()) {
                adjList[v] = Vertex{v, {}};
            }
            adjList[u].neighbors.push_back(&adjList[v]);
            adjList[v].neighbors.push_back(&adjList[u]);
            numEdges++;
        }
    
        numVertices = adjList.size();
        file.close();
    }  
    
    bool addEdge(int u, int v, float weight = 1.0f) {
        // Check if vertices exist
        if (adjList.find(u) == adjList.end()) {
            adjList[u] = Vertex{u, {}};
            numVertices++;
        }
        if (adjList.find(v) == adjList.end()) {
            adjList[v] = Vertex{v, {}};
            numVertices++;
        }

        // Check if edge already exists
        for (auto& neighbor : adjList[u].neighbors) {
            if (neighbor->id == v) {
                return false; // Edge already exists
            }
        }

        // Add edge in both directions (undirected graph)
        adjList[u].neighbors.push_back(&adjList[v]);
        adjList[v].neighbors.push_back(&adjList[u]);
        numEdges++;
        
        return true;
    }

    bool removeEdge(int u, int v) {
        // Check if vertices exist
        if (adjList.find(u) == adjList.end() || adjList.find(v) == adjList.end()) {
            return false;
        }

        // Remove v from u's neighbors
        auto& uNeighbors = adjList[u].neighbors;
        bool edgeFound = false;
        
        for (auto it = uNeighbors.begin(); it != uNeighbors.end(); ) {
            if ((*it)->id == v) {
                it = uNeighbors.erase(it);
                edgeFound = true;
            } else {
                ++it;
            }
        }

        // Remove u from v's neighbors
        auto& vNeighbors = adjList[v].neighbors;
        for (auto it = vNeighbors.begin(); it != vNeighbors.end(); ) {
            if ((*it)->id == u) {
                it = vNeighbors.erase(it);
            } else {
                ++it;
            }
        }

        if (edgeFound) {
            numEdges--;
            return true;
        }
        
        return false;
    }
    
    map<int, Vertex>& getAdjList() {
        return adjList;
    }

    void displayGraph() {
        cout << "Number of vertices: " << numVertices << endl;
        cout << "Number of edges: " << numEdges / 2 << endl;
    }
};

class Dijkstra {
private:
    map<int, Vertex>& graph;
    map<int, int> predecessors;
    map<int, double> distances;
    int source;

    // Comparator for priority queue
    struct VertexDistancePair {
        int vertexId;
        double distance;

        VertexDistancePair(int v, double d) : vertexId(v), distance(d) {}

        // For priority queue (min heap)
        bool operator>(const VertexDistancePair& other) const {
            return distance > other.distance;
        }
    };

public:
    Dijkstra(map<int, Vertex>& g) : graph(g) {}

    void setGraph(map<int, Vertex>& g) {
        graph = g;
    }

    void initialize(int sourceId) {
        source = sourceId;
        
        // Reset prior results
        predecessors.clear();
        distances.clear();
        
        // Initialize distances to infinity and predecessors to -1
        for (const auto& pair : graph) {
            int vertexId = pair.first;
            distances[vertexId] = numeric_limits<double>::infinity();
            predecessors[vertexId] = -1;
        }
        
        // Set source distance to 0
        distances[source] = 0.0;
    }

    void run() {
        // Priority queue to store vertices that need to be processed
        priority_queue<VertexDistancePair, vector<VertexDistancePair>, greater<VertexDistancePair>> pq;
        map<int, bool> visited;
        
        // Initialize visited flags
        for (const auto& pair : graph) {
            visited[pair.first] = false;
        }
        
        // Start with source vertex
        pq.push(VertexDistancePair(source, 0.0));
        
        while (!pq.empty()) {
            // Extract vertex with minimum distance
            VertexDistancePair current = pq.top();
            pq.pop();
            
            int u = current.vertexId;
            
            // If already processed, skip
            if (visited[u]) continue;
            visited[u] = true;
            
            // Process all neighbors
            for (Vertex* neighbor : graph[u].neighbors) {
                int v = neighbor->id;
                
                double weight = graph[u].weight;
                
                // Relaxation step
                if (!visited[v] && distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    predecessors[v] = u;
                    pq.push(VertexDistancePair(v, distances[v]));
                }
            }
        }
    }
    
    const map<int, double>& get_distances() const {
        return distances;
    }
    
    const map<int, int>& get_predecessors() const {
        return predecessors;
    }
    
    // Get the shortest path from source to a specific destination
    vector<int> getPath(int destination) const {
        vector<int> path;
        
        // Check if destination is reachable
        if (distances.find(destination) == distances.end() || 
            distances.at(destination) == numeric_limits<double>::infinity()) {
            return path; // Empty path, no route exists
        }
        
        // Reconstruct path by backtracking from destination to source
        for (int at = destination; at != -1; at = predecessors.at(at)) {
            path.push_back(at);
        }
        
        // Reverse to get path from source to destination
        reverse(path.begin(), path.end());
        return path;
    }
    
    // Get the shortest distance to a specific destination
    double getDistance(int destination) const {
        if (distances.find(destination) == distances.end()) {
            return numeric_limits<double>::infinity();
        }
        return distances.at(destination);
    }
    
    // Print all shortest paths from source
    void printAllPaths() const {
        cout << "Shortest paths from vertex " << source << ":" << endl;
        
        for (const auto& pair : distances) {
            int destination = pair.first;
            double distance = pair.second;
            
            cout << "To vertex " << destination << " (distance: " << distance << "): ";
            
            if (distance == numeric_limits<double>::infinity()) {
                cout << "No path exists" << endl;
                continue;
            }
            
            vector<int> path = getPath(destination);
            for (size_t i = 0; i < path.size(); i++) {
                cout << path[i];
                if (i < path.size() - 1) cout << " -> ";
            }
            cout << endl;
        }
    }

    void updateSSSP(vector<pair<int, int>> insertions, vector<pair<int, int>> deletions) {
        // Data structures for affected vertices
        map<int, bool> affected;
        map<int, bool> affected_del;
        
        // Keep track of edge changes
        vector<pair<int, int>> edgesToProcess;
        vector<pair<int, int>> deletedEdges;
        vector<pair<int, int>> insertedEdges;
        
        // Initialize data structures
        for (const auto& pair : graph) {
            int vertexId = pair.first;
            affected[vertexId] = false;
            affected_del[vertexId] = false;
        }
        
        // Algorithm 2: Identify Affected Vertices
        // Process deleted edges
        for (const auto& edge : deletions) {
            int u = edge.first;
            int v = edge.second;
            
            // Check if edge is in the shortest path tree
            if (predecessors[v] == u || predecessors[u] == v) {
                // Determine which vertex to mark as affected
                int y;
                if (distances[u] > distances[v]) {
                    y = u;
                } else {
                    y = v;
                }
                
                // Set distance to infinity and mark as affected
                distances[y] = numeric_limits<double>::infinity();
                affected_del[y] = true;
                affected[y] = true;
                
                // Mark edge as deleted (for bookkeeping)
                deletedEdges.push_back(edge);
            }
        }
        
        // Process inserted edges
        for (const auto& edge : insertions) {
            int u = edge.first;
            int v = edge.second;
            
            // Determine x based on which vertex has higher distance
            int x, y;
            if (distances[u] > distances[v]) {
                x = v;
                y = u;
            } else {
                x = u;
                y = v;
            }
            
            // Check if the new edge provides a shorter path
            if (distances[y] > distances[x] + graph[x].weight) {
                distances[y] = distances[x] + graph[x].weight;
                predecessors[y] = x;
                affected[y] = true;
                
                // Add edge to the graph (if not already done)
                insertedEdges.push_back(edge);
            }
        }
        
        // Algorithm 3: Update Affected Vertices
        // First handle vertices affected by deletions
        while (true) {
            bool hasAffectedDel = false;
            
            // Check if there are any affected vertices
            for (auto& pair : affected_del) {
                if (pair.second) {
                    hasAffectedDel = true;
                    break;
                }
            }
            
            if (!hasAffectedDel) break;
            
            // Process each affected vertex (would be in parallel in the algorithm)
            map<int, bool> toProcess;
            for (auto& pair : affected_del) {
                if (pair.second) {
                    int v = pair.first;
                    toProcess[v] = true;
                    pair.second = false; // Reset for next iteration
                }
            }
            
            for (const auto& pair : toProcess) {
                int v = pair.first;
                
                // Find all children of v in the SSSP tree
                for (const auto& graphPair : graph) {
                    int c = graphPair.first;
                    
                    // Check if c is a child of v in the tree
                    if (predecessors[c] == v) {
                        // Set distance to infinity
                        distances[c] = numeric_limits<double>::infinity();
                        affected_del[c] = true;
                        affected[c] = true;
                    }
                }
            }
        }
        
        // Now handle all affected vertices
        while (true) {
            bool hasAffected = false;
            
            // Check if there are any affected vertices
            for (auto& pair : affected) {
                if (pair.second) {
                    hasAffected = true;
                    break;
                }
            }
            
            if (!hasAffected) break;
            
            // Process each affected vertex (would be in parallel in the algorithm)
            map<int, bool> toProcess;
            for (auto& pair : affected) {
                if (pair.second) {
                    int v = pair.first;
                    toProcess[v] = true;
                    pair.second = false; // Reset for next iteration
                }
            }
            
            for (const auto& pair : toProcess) {
                int v = pair.first;
                
                // Check all neighbors of v
                for (Vertex* neighbor : graph[v].neighbors) {
                    int n = neighbor->id;
                    double weight = graph[v].weight;
                    
                    // Check if we can improve the path to n through v
                    if (distances[n] > distances[v] + weight) {
                        distances[n] = distances[v] + weight;
                        predecessors[n] = v;
                        affected[n] = true;
                    }
                    // Or vice versa (as in Algorithm 3)
                    else if (distances[v] > distances[n] + weight) {
                        distances[v] = distances[n] + weight;
                        predecessors[v] = n;
                        affected[v] = true;
                    }
                }
            }
        }
        
        // Special handling for source vertex, ensure it stays at distance 0
        distances[source] = 0.0;
        predecessors[source] = -1;
    }
};

int main() {
    Graph graph;
    graph.buildAdjacencyList("../../graphs/initial_graph.txt");
    map<int, Vertex> results = graph.getAdjList();

    Dijkstra dijkstra(results);
    dijkstra.initialize(1);
    dijkstra.run();
    
    const auto& distances = dijkstra.get_distances();
    const auto& predecessors = dijkstra.get_predecessors();
    
    vector<pair<int, int>> insertions = {
        {6, 11},    // Connects mid-level nodes (original 5-10)
        {9, 17},    // Creates a shortcut (original 8-16)
        {4, 8},     // Cross-branch link (original 3-7)
        {1, 11},    // Root-to-mid-level jump (original 0-10)
        {13, 25}    // Deep-level connection (original 12-24)
    };
    
    vector<pair<int, int>> deletions = {
        {1, 6},     // Root-level edge removal (original 0-5)
        {11, 15},   // Chain breaker (original 10-14)
        {3, 8},     // Alternative path removal (original 2-7)
        {7, 11},    // Mid-level connection (original 6-10)
        {17, 21}    // Deep-level removal (original 16-20)
    };

    // Start timing for TOTAL update process
    auto total_start = chrono::high_resolution_clock::now();

    // Edge modifications
    for (const auto& insertion : insertions) {
        int u = insertion.first;
        int v = insertion.second;
        graph.addEdge(u, v);
    }

    for (const auto& deletion : deletions) {
        int u = deletion.first;
        int v = deletion.second;
        graph.removeEdge(u, v);
    }

    // Get updated graph
    results = graph.getAdjList();
    dijkstra.setGraph(results);

    // Start timing for JUST the SSSP update
    auto sssp_start = chrono::high_resolution_clock::now();
    
    // Perform SSSP update
    dijkstra.updateSSSP(insertions, deletions);
    
    // End timing for SSSP update
    auto sssp_end = chrono::high_resolution_clock::now();
    
    // End timing for TOTAL update
    auto total_end = chrono::high_resolution_clock::now();

    // Calculate durations
    auto total_duration = chrono::duration_cast<chrono::microseconds>(total_end - total_start);
    auto sssp_duration = chrono::duration_cast<chrono::microseconds>(sssp_end - sssp_start);

    // Print timing results
    cout << "\nPerformance Metrics:\n";
    cout << "Total update time (edge modifications + SSSP): " << total_duration.count() << " μs\n";
    cout << "SSSP update time only: " << sssp_duration.count() << " μs\n";
    cout << "Edge modification time: " << (total_duration.count() - sssp_duration.count()) << " μs\n";

    // Print results with predecessors
    // for (const auto& vertex : results) {
    //     int vertexId = vertex.first;
    //     cout << "Vertex " << vertexId << ": distance = " << distances.at(vertexId);
    //     if (predecessors.find(vertexId) != predecessors.end()) {
    //         cout << ", predecessor = " << predecessors.at(vertexId);
    //     }
    //     cout << endl;
    // }
    
    return 0;
}