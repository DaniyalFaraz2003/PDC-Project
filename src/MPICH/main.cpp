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
#include <queue>
#include <unordered_map>
#include <limits>
#include <algorithm>

using namespace std;

struct Vertex {
    int id;
    int partitionId;
    vector<Vertex> neighbors;
    vector<float> weights;
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
        ofstream outfile("../../metis_graph/output.graph");
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
        string fileName = "../../metis_graph/output.graph";
        string command = "gpmetis " + fileName + " " + to_string(size);
        int result = system(command.c_str());
        if (result != 0) {
            cerr << "Error running gpmetis!" << endl;
            return;
        }
        cout << "Graph partitioning completed." << endl;
    }

    void mergeOutputGraphs(int size) {
        string file1 = "../../metis_graph/output.graph";
        string file2 = "../../metis_graph/output.graph.part." + to_string(size);
        string mergedFile = "../../metis_graph/merged_file.graph";
    
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
        string file = "../../../../metis_graph/merged_file.graph";
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
                vertex.weights.push_back(1.0f);
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

class ParallelDijkstra {
private:
    int rank;
    int world_size;
    vector<Vertex>& vertices; // Reference to local vertices
    unordered_map<int, float> distance; // Distance map (vertex ID -> distance)
    unordered_map<int, int> predecessor; // Predecessor map for path reconstruction
    map<int, int> vertex_to_partition; // Maps vertex IDs to their partition/rank

    // Priority queue element: (distance, vertex_id)
    using QueueElement = pair<float, int>;
    priority_queue<QueueElement, vector<QueueElement>, greater<QueueElement>> pq;

    // Track pending MPI requests for asynchronous communication
    vector<MPI_Request> pending_requests;

    // Change send_distance_update to use the correct types
    void send_distance_update(int vertex_id, float new_dist) {
        auto it = vertex_to_partition.find(vertex_id);
        if (it == vertex_to_partition.end()) {
            // Don't know where to send this - maybe log a warning
            return;
        }

        int dest_rank = it->second;
        if (dest_rank == rank) {
            // Shouldn't happen - we checked before calling
            return;
        }

        // Pack vertex_id and distance correctly
        vector<char> buffer(sizeof(int) + sizeof(float));
        int* id_ptr = reinterpret_cast<int*>(buffer.data());
        float* dist_ptr = reinterpret_cast<float*>(buffer.data() + sizeof(int));
        
        *id_ptr = vertex_id;
        *dist_ptr = new_dist;

        // Send asynchronously
        MPI_Request req;
        MPI_Isend(buffer.data(), buffer.size(), MPI_CHAR, dest_rank, 0, MPI_COMM_WORLD, &req);
        pending_requests.push_back(req);
    }

    // Update process_incoming_messages to match the new format
    void process_incoming_messages() {
        int flag = 1;
        MPI_Status status;
        
        // Check for messages
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        while (flag) {
            int count;
            MPI_Get_count(&status, MPI_CHAR, &count);
            
            // Each message contains a pair of (vertex_id, distance)
            vector<char> buffer(count);
            MPI_Recv(buffer.data(), count, MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Unpack data safely
            int vertex_id = *reinterpret_cast<int*>(buffer.data());
            float new_dist = *reinterpret_cast<float*>(buffer.data() + sizeof(int));
            
            // Update distance if improved
            auto dist_it = distance.find(vertex_id);
            if (dist_it != distance.end() && new_dist < dist_it->second) {
                distance[vertex_id] = new_dist;
                
                // Add to priority queue if it's a local vertex
                auto part_it = vertex_to_partition.find(vertex_id);
                if (part_it != vertex_to_partition.end() && part_it->second == rank) {
                    pq.push({new_dist, vertex_id});
                }
            }
            
            // Check for next message
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        }
    }

public:
    ParallelDijkstra(int rank, int world_size, vector<Vertex>& vertices, const map<int, int>& vertex_partition_map)
        : vertices(vertices) {
        this->rank = rank;
        this->world_size = world_size;
        this->vertex_to_partition = vertex_partition_map;

        // Initialize all KNOWN vertices with infinity
        for (const auto& vertex : vertices) {
            distance[vertex.id] = numeric_limits<float>::infinity();
            
            // Also initialize distances for neighbor vertices we know about
            for (const auto& neighbor : vertex.neighbors) {
                if (distance.find(neighbor.id) == distance.end()) {
                    distance[neighbor.id] = numeric_limits<float>::infinity();
                }
            }
        }
    }

    ~ParallelDijkstra() {
        // Clean up any pending MPI requests
        for (auto& req : pending_requests) {
            if (req != MPI_REQUEST_NULL) {
                MPI_Request_free(&req);
            }
        }
    }

    void initialize(int source_vertex) {        
        // Set source vertex distance to 0
        auto src_part_it = vertex_to_partition.find(source_vertex);
        if (src_part_it != vertex_to_partition.end() && src_part_it->second == rank) {
            // Source is local - set distance to 0
            distance[source_vertex] = 0.0f;
            pq.push({0.0f, source_vertex});
            cout << "Rank " << rank << " initialized source vertex " << source_vertex << " with distance 0" << endl;
        } else if (distance.find(source_vertex) != distance.end()) {
            // Source is known but not local - initialize with infinity
            // We'll receive updates from the process that owns it
            distance[source_vertex] = numeric_limits<float>::infinity();
        }
        
        // Global synchronization to ensure the source is initialized somewhere
        int have_source = (src_part_it != vertex_to_partition.end() && src_part_it->second == rank) ? 1 : 0;
        int global_have_source = 0;
        MPI_Allreduce(&have_source, &global_have_source, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        if (global_have_source == 0 && rank == 0) {
            cerr << "WARNING: Source vertex " << source_vertex << " not found in any partition!" << endl;
        }
    }

    // Perform one iteration of Dijkstra's algorithm
    bool step() {
        // First, check for any incoming messages
        process_incoming_messages();

        if (pq.empty()) {
            return false; // No more work to do
        }

        auto current = pq.top();
        pq.pop();
        float current_dist = current.first;
        int u = current.second;

        // Find the vertex in our local list (if it exists)
        Vertex* vertex_ptr = nullptr;
        for (auto& v : vertices) {
            if (v.id == u) {
                vertex_ptr = &v;
                break;
            }
        }

        // If we don't have this vertex locally, skip (it was just a distance update)
        if (!vertex_ptr) {
            return true;
        }

        // Relax all edges
        for (size_t i = 0; i < vertex_ptr->neighbors.size(); ++i) {
            int v = vertex_ptr->neighbors[i].id;
            float weight = vertex_ptr->weights[i];
            float new_dist = current_dist + weight;

            // Check if we need to update the distance
            if (new_dist < distance[v]) {
                distance[v] = new_dist;
                predecessor[v] = u;

                // Check if the neighbor is local or remote
                auto it = vertex_to_partition.find(v);
                if (it != vertex_to_partition.end() && it->second == rank) {
                    // Local vertex - add to priority queue
                    pq.push({new_dist, v});
                } else {
                    // Remote vertex - send update to owning process
                    send_distance_update(v, new_dist);
                }
            }
        }

        return true;
    }

    void run() {
        int local_done = 0;
        int global_done = 0;
        
        do {
            // Process any incoming messages first
            process_incoming_messages();
            
            // Process local queue
            if (!pq.empty()) {
                step();
                local_done = 0;  // Still have work to do
            } else {
                local_done = 1;  // Local work is done for now
            }
            
            // Wait for all pending communications to complete
            if (!pending_requests.empty()) {
                MPI_Waitall(pending_requests.size(), pending_requests.data(), MPI_STATUSES_IGNORE);
                pending_requests.clear();
            }
            
            // Check if all processes are done
            MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            
            // All processes are done if global_done equals world_size
            // But we need one more round to ensure all messages are processed
            if (global_done == world_size) {
                // Process any final messages
                process_incoming_messages();
                
                // Recheck if we're still done
                local_done = pq.empty() ? 1 : 0;
                MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
            
        } while (global_done < world_size);
        
        // Debug output - counts for non-infinity distances
        int non_inf_count = 0;
        for (const auto& [id, dist] : distance) {
            if (dist < numeric_limits<float>::infinity()) {
                non_inf_count++;
            }
        }
        cout << "Rank " << rank << " finished with " << non_inf_count << " reachable vertices" << endl;
    }

    // Get the computed distances (only valid for local vertices)
    const unordered_map<int, float>& get_distances() const {
        return distance;
    }

    // Get the computed predecessors (only valid for local vertices)
    const unordered_map<int, int>& get_predecessors() const {
        return predecessor;
    }

    // Algorithm 1: Updating SSSP for a Single Change
    void single_change_update(int u, int v, float new_weight, bool is_inserted) {
        // Find the affected vertex x
        int x;
        if (distance.find(u) == distance.end() || distance.find(v) == distance.end()) {
            // One of the vertices is not known to this process
            return;
        }
        
        if (distance[u] > distance[v]) {
            x = v;
        } else {
            x = u;
        }
        
        // Initialize Priority Queue with x
        priority_queue<QueueElement, vector<QueueElement>, greater<QueueElement>> local_pq;
        local_pq.push({distance[x], x});
        
        // Update distance for x if edge is inserted
        if (is_inserted) {
            if (x == v && distance[u] + new_weight < distance[v]) {
                distance[v] = distance[u] + new_weight;
                predecessor[v] = u;
                local_pq.push({distance[v], v});
            } else if (x == u && distance[v] + new_weight < distance[u]) {
                distance[u] = distance[v] + new_weight;
                predecessor[u] = v;
                local_pq.push({distance[u], u});
            }
        } else { // Edge is deleted
            // For deletion, we need to identify all affected vertices
            // We'll call identify_affected_vertices function
            vector<int> affected_vertices = identify_affected_vertices(u, v);
            
            // Reset distances for affected vertices to infinity
            for (int vertex : affected_vertices) {
                if (vertex_to_partition[vertex] == rank) {
                    distance[vertex] = numeric_limits<float>::infinity();
                    local_pq.push({distance[vertex], vertex});
                }
            }
        }
        
        // Process the queue to update affected subgraph
        while (!local_pq.empty()) {
            auto current = local_pq.top();
            local_pq.pop();
            int z = current.second;
            float current_dist = current.first;
            
            // Find the vertex in our local list
            bool is_local = false;
            Vertex* vertex_ptr = nullptr;
            for (auto& v : vertices) {
                if (v.id == z) {
                    vertex_ptr = &v;
                    is_local = true;
                    break;
                }
            }
            
            if (!is_local) {
                continue; // Skip non-local vertices
            }
            
            bool updated = false;
            
            // Relax all edges from z
            for (size_t i = 0; i < vertex_ptr->neighbors.size(); ++i) {
                int neighbor = vertex_ptr->neighbors[i].id;
                float weight = vertex_ptr->weights[i];
                float new_dist = current_dist + weight;
                
                if (new_dist < distance[neighbor]) {
                    distance[neighbor] = new_dist;
                    predecessor[neighbor] = z;
                    updated = true;
                    
                    // Check if neighbor is local or remote
                    auto it = vertex_to_partition.find(neighbor);
                    if (it != vertex_to_partition.end() && it->second == rank) {
                        // Local vertex - add to priority queue
                        local_pq.push({new_dist, neighbor});
                    } else {
                        // Remote vertex - send update to owning process
                        send_distance_update(neighbor, new_dist);
                    }
                }
            }
        }
    }

    // Algorithm 2: Identify Affected Vertices
    vector<int> identify_affected_vertices(int u, int v) {
        vector<int> affected_vertices;
        unordered_map<int, bool> affected_map;
        
        // Initialize affected vertices
        affected_map[v] = true;
        affected_vertices.push_back(v);
        
        // For each vertex in our partition
        for (auto& vertex : vertices) {
            int vertex_id = vertex.id;
            
            // Check if this vertex is affected by the edge deletion
            if (predecessor[vertex_id] == u && vertex_id == v) {
                // This vertex used the deleted edge directly
                distance[vertex_id] = numeric_limits<float>::infinity();
                affected_map[vertex_id] = true;
                affected_vertices.push_back(vertex_id);
            }
            
            // Process affected vertices in a BFS manner
            for (size_t i = 0; i < affected_vertices.size(); i++) {
                int affected_id = affected_vertices[i];
                
                // Find children of this affected vertex
                for (auto& v : vertices) {
                    if (predecessor[v.id] == affected_id) {
                        // This vertex is a child of an affected vertex
                        distance[v.id] = numeric_limits<float>::infinity();
                        if (!affected_map[v.id]) {
                            affected_map[v.id] = true;
                            affected_vertices.push_back(v.id);
                        }
                    }
                }
            }
        }
        
        return affected_vertices;
    }

    // Algorithm 3: Update Affected Vertices
    void update_affected_vertices(const vector<int>& affected_vertices) {
        // Create a priority queue with all non-affected vertices
        priority_queue<QueueElement, vector<QueueElement>, greater<QueueElement>> local_pq;
        
        // Add all vertices with known distances that weren't affected
        for (const auto& [id, dist] : distance) {
            if (dist < numeric_limits<float>::infinity() && 
                find(affected_vertices.begin(), affected_vertices.end(), id) == affected_vertices.end()) {
                local_pq.push({dist, id});
            }
        }
        
        // Process the queue to update affected vertices
        while (!local_pq.empty()) {
            auto current = local_pq.top();
            local_pq.pop();
            int z = current.second;
            float current_dist = current.first;
            
            // Find the vertex in our local list
            bool is_local = false;
            Vertex* vertex_ptr = nullptr;
            for (auto& v : vertices) {
                if (v.id == z) {
                    vertex_ptr = &v;
                    is_local = true;
                    break;
                }
            }
            
            if (!is_local) {
                continue; // Skip non-local vertices
            }
            
            // Relax all edges from z
            for (size_t i = 0; i < vertex_ptr->neighbors.size(); ++i) {
                int neighbor = vertex_ptr->neighbors[i].id;
                float weight = vertex_ptr->weights[i];
                float new_dist = current_dist + weight;
                
                // Check if we're improving the distance
                if (new_dist < distance[neighbor]) {
                    distance[neighbor] = new_dist;
                    predecessor[neighbor] = z;
                    
                    // Check if neighbor is local or remote
                    auto it = vertex_to_partition.find(neighbor);
                    if (it != vertex_to_partition.end() && it->second == rank) {
                        // Local vertex - add to priority queue
                        local_pq.push({new_dist, neighbor});
                    } else {
                        // Remote vertex - send update to owning process
                        send_distance_update(neighbor, new_dist);
                    }
                }
            }
        }
    }

    // Main dynamic SSSP update function
    void update_edge(int u, int v, float new_weight, bool is_inserted) {
        // First, synchronize with all processes
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            cout << "Updating edge (" << u << ", " << v << ") with weight " 
                << new_weight << ", operation: " << (is_inserted ? "insert" : "delete") << endl;
        }
        
        // For edge insertion, use single_change_update directly
        if (is_inserted) {
            single_change_update(u, v, new_weight, true);
        } else {
            // For edge deletion, we need to:
            // 1. Identify affected vertices
            vector<int> affected_vertices = identify_affected_vertices(u, v);
            
            // 2. Update affected vertices
            update_affected_vertices(affected_vertices);
        }
        
        // Wait for all pending communications to complete
        if (!pending_requests.empty()) {
            MPI_Waitall(pending_requests.size(), pending_requests.data(), MPI_STATUSES_IGNORE);
            pending_requests.clear();
        }
        
        // Synchronize again once update is complete
        MPI_Barrier(MPI_COMM_WORLD);
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
                
                vertex.weights.push_back(1.0f);
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
        graph.loadGraphFromFile("../../graphs/initial_graph.txt");
        graph.convertGraphToMetisGraph();
        graph.applyMetisPartitioning(size);
        graph.mergeOutputGraphs(size);

        ifstream file("../../metis_graph/merged_file.graph");
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
    MPI_Barrier(MPI_COMM_WORLD);
    // displayVertexList(listOfVertices);


    bool result = testPartition(listOfVertices, rank, size);

    cout << "Rank " << rank << " has " << (result ? "valid" : "invalid") << " partitions." << endl;

    MPI_Barrier(MPI_COMM_WORLD);

    cout << "Rank " << rank << " is applying Dijkstra Algorithm " << endl;

    ParallelDijkstra dijkstra(rank, size, listOfVertices, vertexPartitions);
    dijkstra.initialize(1); // Initialize with source vertex ID 1
    dijkstra.run();

    // Example edge update - insert a new edge
    // int u = 120, v = 121;
    // float weight = 0.5;
    // bool is_inserted = true;
    // dijkstra.update_edge(u, v, weight, is_inserted);
    
    // // Example edge update - delete an edge
    // u = 3;
    // v = 4;
    // is_inserted = false;
    // dijkstra.update_edge(u, v, 0.0, is_inserted);
    
    const auto& distances = dijkstra.get_distances();
    const auto& predecessors = dijkstra.get_predecessors();

    // Print local results
    for (const auto& vertex : listOfVertices) {
        cout << "Vertex " << vertex.id << ": distance = " << distances.at(vertex.id);
        if (predecessors.find(vertex.id) != predecessors.end()) {
            cout << ", predecessor = " << predecessors.at(vertex.id);
        }
        cout << endl;
    }
    
    MPI_Finalize();

    return 0;
}
