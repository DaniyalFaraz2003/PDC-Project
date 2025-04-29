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

    void mergeOutputGraphs() {
        string file1 = "../metis_graph/output.graph";
        string file2 = "../metis_graph/output.graph.part.2";
        string mergedFile = "../metis_graph/merged_file.graph";
    
        ifstream inFile1(file1);
        ifstream inFile2(file2);
        ofstream outFile(mergedFile);
    
        if (!inFile1.is_open() || !inFile2.is_open() || !outFile.is_open()) {
            cerr << "Error opening one of the files!" << endl;
            return;
        }
    
        string line;
        
        // Skipping first line of file1 (contains number of vertices and edges)
        getline(inFile1, line);
    
        int vertexIndex = 1;
        while (getline(inFile1, line)) {
            string partitionStr;
            if (!getline(inFile2, partitionStr)) {
                cerr << "Mismatch in number of lines between graph and partition file!" << endl;
                break;
            }
    
            int partition = stoi(partitionStr);
    
            // Writing "partition vertex neighbors" in merged_file
            outFile << partition << " " << vertexIndex << " " << line << endl;
    
            vertexIndex++;
        }
    
        inFile1.close();
        inFile2.close();
        outFile.close();
    
        cout << "Merged file written to: " << mergedFile << endl;
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
    graph.mergeOutputGraphs();
    graph.displayGraph();

    // MPI_Init(&argc, &argv);

    // int rank, size;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI_Finalize();

    return 0;
}