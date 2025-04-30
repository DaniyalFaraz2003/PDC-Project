#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <mpi.h>
#include <cstring>
#include <unordered_set>
#include <set>

using namespace std;

struct Vertex {
    int id;
    int partitionId;
    vector<Vertex> neighbors;
};

class Graph {
private:
    map<int, vector<int>> adjList;
    map<int, int> vertexPartition;
    vector<Vertex> listOfVertices;
    int numVertices;
    int numEdges;

public:
    Graph() {
        numVertices = 0;
        numEdges = 0;
    }

    int getNumVertices() { return numVertices; }

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

        // Clear the adjList to free memory
        adjList.clear();
    }    
    
    void applyMetisPartitioning(int size) {
        string fileName = "../metis_graph/output.graph";
        string command = "gpmetis " + fileName + " " + to_string(size);
        int result = system(command.c_str());
        if (result != 0) {
            cerr << "Error running gpmetis!" << endl;
            return;
        }
        cout << "Graph partitioning completed." << endl;
    }

    void mergeOutputGraphs(int size) {
        string file1 = "../metis_graph/output.graph";
        string file2 = "../metis_graph/output.graph.part." + to_string(size);
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

    void buildListOfVertices() {
        string file = "../metis_graph/merged_file.graph";
        ifstream inFile(file);
    
        if (!inFile.is_open()) {
            cerr << "Error opening file: " << file << endl;
            return;
        }
    
        string line;
        while (getline(inFile, line)) {
            istringstream iss(line);
            int partition, vertexId;
            iss >> partition >> vertexId;
    
            Vertex vertex;
            vertex.id = vertexId;
            vertex.partitionId = partition;
    
            int neighborId;
            while (iss >> neighborId) {
                Vertex neighbor;
                neighbor.id = neighborId;
                neighbor.partitionId = vertexPartition[neighborId]; // using your partition map
                vertex.neighbors.push_back(neighbor);
            }
    
            listOfVertices.push_back(vertex);
        }
    
        inFile.close();
    
        cout << "Vertex list constructed with " << listOfVertices.size() << " vertices." << endl;
    }
    
    void displayVertexList() {
        cout << "\n--- Displaying First 10 Vertices ---\n";
        cout << "Partition | Vertex ID | Neighbors\n";
        cout << "--------------------------------------------\n";
    
        int count = 0;
        for (const auto& vertex : listOfVertices) {
            if (count >= 10) break;
    
            cout << "    " << vertex.partitionId
                 << "     |     " << vertex.id << "     | ";
    
            for (const auto& neighbor : vertex.neighbors) {
                cout << neighbor.id << " ";
            }
            cout << endl;
    
            count++;
        }
    
        cout << "--------------------------------------------\n";
    }    
    
    void displayGraph() {
        cout << "Number of vertices: " << numVertices << endl;
        cout << "Number of edges: " << numEdges / 2 << endl;
    }
};

map<int, int> buildVertexPartition(vector<string>& lines) {
    map<int, int> vertexPartition;

    for (const auto& line : lines) {
        istringstream iss(line);
        int partition, vertexId;
        if (!(iss >> partition >> vertexId)) {
            cerr << "Error parsing line: " << line << endl;
            continue;
        }
        vertexPartition[vertexId] = partition;
    }

    return vertexPartition;
}

vector<Vertex> buildListOfVertices(vector<string>& lines, map<int, int>& vertexPartition, int rank) {
    vector<Vertex> listOfVertices;
    set<int> missingPartitionInfo;

    for (const auto& line : lines) {
        istringstream iss(line);
        int partition, vertexId;
        
        if (!(iss >> partition >> vertexId)) {
            cerr << "Error parsing line: " << line << endl;
            continue;
        }

        if (partition == rank) {
            Vertex vertex;
            vertex.id = vertexId;
            vertex.partitionId = partition;

            int neighborId;
            while (iss >> neighborId) {
                Vertex neighbor;
                neighbor.id = neighborId;
                
                // Check if we know this neighbor's partition
                auto it = vertexPartition.find(neighborId);
                if (it != vertexPartition.end()) {
                    neighbor.partitionId = it->second;
                } else {
                    missingPartitionInfo.insert(neighborId);
                    neighbor.partitionId = rank; // Default to our partition as fallback
                }
                
                vertex.neighbors.push_back(neighbor);
            }

            listOfVertices.push_back(vertex);
        }
    }

    // Report any missing partition information
    if (!missingPartitionInfo.empty()) {
        cerr << "Process " << rank << " found " << missingPartitionInfo.size() 
             << " neighbors with unknown partition assignment." << endl;
        
        // Optionally print some examples
        int count = 0;
        for (int id : missingPartitionInfo) {
            if (count++ < 5)
                cerr << "  Missing partition for vertex ID: " << id << endl;
        }
        if (count > 5)
            cerr << "  ... and " << (missingPartitionInfo.size() - 5) << " more" << endl;
    }
    
    return listOfVertices;
}

void displayVertexList(vector<Vertex>& listOfVertices) {
    
    cout << "\n--- Displaying First 5 Vertices ---\n";
    cout << "Partition | Vertex ID | Neighbors\n";
    cout << "--------------------------------------------\n";

    int count = 0;
    for (const auto& vertex : listOfVertices) {
        if (count >= 5) break;

        cout << "    " << vertex.partitionId
             << "     |     " << vertex.id << "     | ";

        for (const auto& neighbor : vertex.neighbors) {
            cout << neighbor.id << " ";
        }
        cout << endl;

        count++;
    }

    cout << "--------------------------------------------\n";
}

bool testPartition(vector<Vertex>& listOfVertices, int rank, int numProcesses) {
    bool allGood = true;
    
    // a quick lookup for local vertex IDs
    unordered_set<int> localVertexIds;
    for (const auto& v : listOfVertices) {
        localVertexIds.insert(v.id);
    }
    
    // Count how many messages each process will receive
    vector<int> sendCounts(numProcesses, 0);
    
    // First, count the messages we'll send to each process
    for (const auto& vertex : listOfVertices) {
        for (const auto& neighbor : vertex.neighbors) {
            if (neighbor.partitionId != rank) {
                sendCounts[neighbor.partitionId]++;
            }
        }
    }
    
    // Share these counts with all processes
    vector<int> recvCounts(numProcesses);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Now everyone knows how many messages to expect from each process
    int totalToReceive = 0;
    for (int count : recvCounts) {
        totalToReceive += count;
    }
    
    // Send the vertex IDs to their respective processes
    vector<MPI_Request> sendRequests;
    for (const auto& vertex : listOfVertices) {
        for (const auto& neighbor : vertex.neighbors) {
            if (neighbor.partitionId != rank) {
                int targetRank = neighbor.partitionId;
                int neighborId = neighbor.id;
                
                MPI_Request req;
                MPI_Isend(&neighborId, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD, &req);
                sendRequests.push_back(req);
            }
        }
    }
    
    // Receive and process all expected messages
    int receivedCount = 0;
    while (receivedCount < totalToReceive) {
        MPI_Status status;
        int vertexId;
        MPI_Recv(&vertexId, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        receivedCount++;
        
        int response = localVertexIds.find(vertexId) != localVertexIds.end();
        MPI_Send(&response, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
    }
    
    // Wait for all our send operations to complete
    MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUS_IGNORE);
    
    // Now receive the validation responses for our queries
    int totalQueries = 0;
    for (int count : sendCounts) {
        totalQueries += count;
    }
    
    for (int i = 0; i < totalQueries; i++) {
        int result;
        MPI_Status status;
        MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        
        if (result != 1) {
            allGood = false;
            cout << "Process " << rank << " found missing vertex on process " << status.MPI_SOURCE << endl;
            break;
        }
    }
     
    return allGood;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    string fileContents;
    int contentLength = 0;
    
    if (rank == 0) {
        Graph graph;
        graph.loadGraphFromFile("../graphs/p2p-Gnutella-small.txt");
        graph.convertGraphToMetisGraph();
        graph.applyMetisPartitioning(size);
        graph.mergeOutputGraphs(size);

        ifstream file("../metis_graph/merged_file.graph");
        if (file.is_open()) {
            stringstream buffer;
            buffer << file.rdbuf();
            fileContents = buffer.str();
            contentLength = fileContents.length();
            file.close();
        } else {
            cerr << "Error opening merged file!" << endl;
        }
    }  

    // Broadcast the length of the file contents to all processes
    MPI_Bcast(&contentLength, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char *buffer = new char[contentLength + 1];
    if (rank == 0) {
        // Copy the file contents to the buffer
        strcpy(buffer, fileContents.c_str());
    }

    // Broadcast the file contents to all processes
    MPI_Bcast(buffer, contentLength + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    buffer[contentLength] = '\0'; // Null-terminate the string

    string receivedContent(buffer);
    delete[] buffer;

    vector<string> lines;   
    istringstream iss(receivedContent);
    string line;
    while (getline(iss, line)) {
        if (!line.empty())
            lines.push_back(line);
    }

    cout << "rank " << rank << " received " << lines.size() << " lines from merged graph" << endl;

    map<int, int> vertexPartitions = buildVertexPartition(lines);
    vector<Vertex> listOfVertices = buildListOfVertices(lines, vertexPartitions, rank);
    // displayVertexList(listOfVertices);

    MPI_Barrier(MPI_COMM_WORLD);

    bool result = testPartition(listOfVertices, rank, size);

    cout << "Rank " << rank << " has " << (result ? "valid" : "invalid") << " partitions." << endl;

    MPI_Finalize();

    return 0;
}