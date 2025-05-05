#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <limits>
#include <fstream>
#include <sstream>
#include <algorithm>
using namespace std;

struct Vertex {
    int id;
    vector<Vertex*> neighbors;
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
                
                // For unweighted graph, weight is 1.0
                double weight = 1.0;
                
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
};

// Example usage in main function
int main(int argc, char** argv) {        
    Graph graph;
    graph.buildAdjacencyList("../../graphs/initial_graph.txt");
    map<int, Vertex> results = graph.getAdjList();

    Dijkstra dijkstra(results);
    dijkstra.initialize(1);
    dijkstra.run();
    
    const auto& distances = dijkstra.get_distances();
    const auto& predecessors = dijkstra.get_predecessors();

    // Print results
    cout << "Distances from source vertex 0:" << endl;
    for (const auto& pair : distances) {
        cout << "To vertex " << pair.first << ": " << pair.second << endl;
    }
    
    return 0;
}