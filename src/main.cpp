#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
// #include <mpi.h>

using namespace std;

class Graph {
private:
    map<int, vector<int>> adjList;
    int numVertices;
    int numEdges;

public:
    Graph() {
        numVertices = 0;
        numEdges = 0;
    }

    void loadGraphFromFile(string fileName) {
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
    
            adjList[u + 1].push_back(v + 1);
            adjList[v + 1].push_back(u + 1);
            numEdges++;
        }
    
        numVertices = adjList.size();
        file.close();
    }        

    void convertGraphToMetisGraph() {
        ofstream outfile("../metis_graph/output.graph");
        if (!outfile.is_open()) {
            cerr << "Error opening output file!" << endl;
            return;
        }
    
        outfile << numVertices << " " << numEdges << endl;
    
        for (int i = 1; i <= numVertices; ++i) {
            if (adjList.find(i) != adjList.end()) {
                for (auto neighbor : adjList[i]) {
                    outfile << neighbor << " ";
                }
            }
            outfile << endl;
        }
    
        outfile.close();
    }    
    
    void applyMetisPartitioning() {
        string fileName = "../metis_graph/output.graph";
        string command = "gpmetis " + fileName + " 2";
        int result = system(command.c_str());
        if (result != 0) {
            cerr << "Error running gpmetis!" << endl;
            return;
        }
        cout << "Graph partitioning completed." << endl;
    }
    
    void displayGraph() {
        cout << "Number of vertices: " << numVertices << endl;
        cout << "Number of edges: " << numEdges / 2 << endl;
    }
};

int main(int argc, char** argv) {
    Graph graph;
    graph.loadGraphFromFile("../graphs/p2p-Gnutella-small.txt");
    graph.convertGraphToMetisGraph();
    graph.applyMetisPartitioning();
    graph.displayGraph();

    // MPI_Init(&argc, &argv);

    // int rank, size;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI_Finalize();

    return 0;
}