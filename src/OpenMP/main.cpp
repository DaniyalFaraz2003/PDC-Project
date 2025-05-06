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
#include <omp.h>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <chrono>

#define NUM_THREADS 256

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
    
        // Read all lines first (sequential I/O is faster)
        vector<string> lines;
        string line;
        while (getline(file, line)) {
            if (!line.empty() && line[0] != '#') {
                lines.push_back(line);
            }
        }
        file.close();
    
        // Temporary storage with one mutex per vertex
        vector<mutex> vertex_mutexes(lines.size() * 2); // Oversized for safety
        adjList.clear();
        numEdges = 0;
    
        #pragma omp parallel for reduction(+:numEdges) schedule(dynamic, NUM_THREADS)
        for (size_t i = 0; i < lines.size(); i++) {
            int u, v;
            istringstream iss(lines[i]);
            if (!(iss >> u >> v)) continue;
    
            u++; v++; // Convert to 1-based indexing
    
            // Lock vertices to prevent concurrent modification
            lock_guard<mutex> lock_u(vertex_mutexes[u % vertex_mutexes.size()]);
            lock_guard<mutex> lock_v(vertex_mutexes[v % vertex_mutexes.size()]);
    
            adjList[u].push_back(v);
            adjList[v].push_back(u);
            numEdges++;
        }
    
        numVertices = adjList.size();
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
        string file = "../../metis_graph/merged_file.graph";
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

    // Add thread-local storage for pending requests
    vector<vector<MPI_Request>> thread_pending_requests;
    mutex pending_mutex;  // Mutex for updating the main pending_requests vector

    void send_distance_update(int vertex_id, float new_dist, int thread_id) {
        auto it = vertex_to_partition.find(vertex_id);
        if (it == vertex_to_partition.end()) {
            return;
        }
        
        int dest_rank = it->second;
        if (dest_rank == rank) {
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
        
        // Store in thread-local pending request list
        thread_pending_requests[thread_id].push_back(req);
    }

    // Parallel processing of incoming messages
    void process_incoming_messages() {
        int flag = 1;
        MPI_Status status;
        
        // Check for messages
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        
        // Collect all messages first to process them in parallel
        vector<vector<char>> message_buffers;
        vector<int> message_sizes;
        
        while (flag) {
            int count;
            MPI_Get_count(&status, MPI_CHAR, &count);
            
            // Each message contains a pair of (vertex_id, distance)
            vector<char> buffer(count);
            MPI_Recv(buffer.data(), count, MPI_CHAR, status.MPI_SOURCE, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            message_buffers.push_back(buffer);
            message_sizes.push_back(count);
            
            // Check for next message
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        }
        
        // Process collected messages in parallel
        if (!message_buffers.empty()) {
            #pragma omp parallel for schedule(dynamic, NUM_THREADS)
            for (size_t i = 0; i < message_buffers.size(); i++) {
                const auto& buffer = message_buffers[i];
                
                // Unpack data safely
                int vertex_id = *reinterpret_cast<const int*>(buffer.data());
                float new_dist = *reinterpret_cast<const float*>(buffer.data() + sizeof(int));
                
                // Update distance if improved - needs atomic operation or critical section
                #pragma omp critical(distance_update)
                {
                    auto dist_it = distance.find(vertex_id);
                    if (dist_it != distance.end() && new_dist < dist_it->second) {
                        distance[vertex_id] = new_dist;
                        
                        // Add to priority queue if it's a local vertex
                        auto part_it = vertex_to_partition.find(vertex_id);
                        if (part_it != vertex_to_partition.end() && part_it->second == rank) {
                            pq.push({new_dist, vertex_id});
                        }
                    }
                }
            }
        }
    }

public:
    ParallelDijkstra(int rank, int world_size, vector<Vertex>& vertices, const map<int, int>& vertex_partition_map)
        : vertices(vertices) {
        this->rank = rank;
        this->world_size = world_size;
        this->vertex_to_partition = vertex_partition_map;
        
        // Initialize thread-local storage for pending requests
        int max_threads = omp_get_max_threads();
        thread_pending_requests.resize(max_threads);
        
        // Initialize distance map with infinity - can be parallelized
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < vertices.size(); i++) {
            const auto& vertex = vertices[i];
            #pragma omp critical(distance_init)
            {
                distance[vertex.id] = numeric_limits<float>::infinity();
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
        // Initialize all KNOWN vertices with infinity - can be parallelized
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 64)
            for (size_t i = 0; i < vertices.size(); i++) {
                const auto& vertex = vertices[i];
                
                #pragma omp critical(distance_init)
                {
                    distance[vertex.id] = numeric_limits<float>::infinity();
                    
                    // Also initialize distances for neighbor vertices we know about
                    for (const auto& neighbor : vertex.neighbors) {
                        if (distance.find(neighbor.id) == distance.end()) {
                            distance[neighbor.id] = numeric_limits<float>::infinity();
                        }
                    }
                }
            }
        }
        
        // Set source vertex distance to 0
        auto src_part_it = vertex_to_partition.find(source_vertex);
        if (src_part_it != vertex_to_partition.end() && src_part_it->second == rank) {
            // Source is local - set distance to 0
            distance[source_vertex] = 0.0f;
            pq.push({0.0f, source_vertex});
            cout << "Rank " << rank << " initialized source vertex " << source_vertex << " with distance 0" << endl;
        } else if (distance.find(source_vertex) != distance.end()) {
            // Source is known but not local - initialize with infinity
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

    // Perform one iteration of Dijkstra's algorithm with OpenMP enhancements
    bool step() {
        // First, check for any incoming messages
        process_incoming_messages();
        
        if (pq.empty()) {
            return false; // No more work to do
        }
        
        // Note: We cannot directly parallelize the priority queue extraction
        // But we can parallelize the edge relaxation after extracting a vertex
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
        
        // Create thread-local updates to avoid contention on the priority queue
        int max_threads = omp_get_max_threads();
        vector<vector<pair<int, float>>> thread_updates(max_threads);
        
        // Relax all edges in parallel
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            
            #pragma omp for schedule(dynamic, 32)
            for (size_t i = 0; i < vertex_ptr->neighbors.size(); ++i) {
                int v = vertex_ptr->neighbors[i].id;
                float weight = vertex_ptr->weights[i];
                float new_dist = current_dist + weight;
                
                // Check if we need to update the distance
                bool need_update = false;
                
                #pragma omp critical(distance_check)
                {
                    if (new_dist < distance[v]) {
                        distance[v] = new_dist;
                        predecessor[v] = u;
                        need_update = true;
                    }
                }
                
                if (need_update) {
                    // Check if the neighbor is local or remote
                    auto it = vertex_to_partition.find(v);
                    if (it != vertex_to_partition.end() && it->second == rank) {
                        // Local vertex - collect updates for priority queue
                        thread_updates[thread_id].push_back({v, new_dist});
                    } else {
                        // Remote vertex - send update to owning process
                        send_distance_update(v, new_dist, thread_id);
                    }
                }
            }
        }
        
        // Merge thread-local updates into the priority queue
        for (const auto& updates : thread_updates) {
            for (const auto& update : updates) {
                pq.push({update.second, update.first});
            }
        }
        
        // Collect all pending requests from threads
        for (int thread_id = 0; thread_id < max_threads; thread_id++) {
            if (!thread_pending_requests[thread_id].empty()) {
                pending_requests.insert(pending_requests.end(), 
                                       thread_pending_requests[thread_id].begin(),
                                       thread_pending_requests[thread_id].end());
                thread_pending_requests[thread_id].clear();
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

        #pragma omp parallel reduction(+:non_inf_count)
        {
            auto it = distance.begin();
            #pragma omp for schedule(static)  // Only static works with forward iterators
            for (size_t i = 0; i < distance.size(); i++) {
                advance(it, 1);  // Manually advance iterator
                if (it->second < numeric_limits<float>::infinity()) {
                    non_inf_count++;
                }
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

    void updateSSSP(vector<pair<int, int>> insertions, vector<pair<int, int>> deletions) {
        // Data structures for affected vertices
        unordered_map<int, bool> affected;
        unordered_map<int, bool> affected_del;
        
        // Initialize data structures for all known vertices
        #pragma omp parallel for
        for (int i = 0; i < distance.size(); i++) {
            auto it = distance.begin();
            advance(it, i);
            int vertex_id = it->first;
            #pragma omp critical
            {
                affected[vertex_id] = false;
                affected_del[vertex_id] = false;
            }
        }
        
        // Vector to track vertices that need updates sent to other processes
        vector<pair<int, float>> updates_to_send;
        
        // Process deleted edges first
        #pragma omp parallel
        {
            vector<pair<int, float>> local_updates;
            
            #pragma omp for nowait
            for (size_t i = 0; i < deletions.size(); i++) {
                int u = deletions[i].first;
                int v = deletions[i].second;
                
                // Skip if we don't know about these vertices
                if (distance.find(u) == distance.end() || distance.find(v) == distance.end()) {
                    continue;
                }
                
                // Check if edge is in the shortest path tree (using predecessor)
                if ((predecessor.find(v) != predecessor.end() && predecessor[v] == u) || 
                    (predecessor.find(u) != predecessor.end() && predecessor[u] == v)) {
                    
                    // Determine which vertex to mark as affected (the one with higher distance)
                    int y = (distance[u] > distance[v]) ? u : v;
                    
                    #pragma omp critical
                    {
                        // Set distance to infinity and mark as affected
                        distance[y] = numeric_limits<float>::infinity();
                        affected_del[y] = true;
                        affected[y] = true;
                        
                        // If this is a local vertex, we'll need to propagate this change
                        auto it = vertex_to_partition.find(y);
                        if (it != vertex_to_partition.end() && it->second == rank) {
                            local_updates.push_back({y, numeric_limits<float>::infinity()});
                        }
                    }
                }
            }
            
            #pragma omp critical
            {
                updates_to_send.insert(updates_to_send.end(), local_updates.begin(), local_updates.end());
            }
        }
        
        // Process inserted edges
        #pragma omp parallel
        {
            vector<pair<int, float>> local_updates;
            
            #pragma omp for nowait
            for (size_t i = 0; i < insertions.size(); i++) {
                int u = insertions[i].first;
                int v = insertions[i].second;
                
                // Skip if we don't know about these vertices
                if (distance.find(u) == distance.end() || distance.find(v) == distance.end()) {
                    continue;
                }
                
                // Find the vertex with lower distance (x) and higher distance (y)
                int x, y;
                if (distance[u] > distance[v]) {
                    x = v;
                    y = u;
                } else {
                    x = u;
                    y = v;
                }
                
                // Find the weight of the edge from x to y
                float weight = 0.0f;
                
                // Try to find vertex x in our local vertices
                for (const auto& vertex : vertices) {
                    if (vertex.id == x) {
                        // Look for neighbor y
                        for (size_t i = 0; i < vertex.neighbors.size(); ++i) {
                            if (vertex.neighbors[i].id == y) {
                                weight = vertex.weights[i];
                                break;
                            }
                        }
                        break;
                    }
                }
                
                // If weight is 0, we might not have found the edge locally
                if (weight == 0.0f) {
                    // For simplicity, we'll use MPI to gather the weight from other processes
                    // This is a simplified approach - a more optimized version would batch these requests
                    float local_weight = weight;
                    MPI_Allreduce(&local_weight, &weight, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                    
                    // If still 0, this edge might not exist in any process
                    if (weight == 0.0f) {
                        continue;
                    }
                }
                
                // Check if the new edge provides a shorter path
                if (distance[y] > distance[x] + weight) {
                    #pragma omp critical
                    {
                        distance[y] = distance[x] + weight;
                        predecessor[y] = x;
                        affected[y] = true;
                        
                        // If this is a local vertex, we'll need to propagate this change
                        auto it = vertex_to_partition.find(y);
                        if (it != vertex_to_partition.end() && it->second == rank) {
                            local_updates.push_back({y, distance[y]});
                        }
                    }
                }
            }
            
            #pragma omp critical
            {
                updates_to_send.insert(updates_to_send.end(), local_updates.begin(), local_updates.end());
            }
        }
        
        // Send initial updates to other processes
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            #pragma omp for
            for (size_t i = 0; i < updates_to_send.size(); i++) {
                int vertex_id = updates_to_send[i].first;
                float new_dist = updates_to_send[i].second;
                
                // For each process that might need this update
                for (int p = 0; p < world_size; ++p) {
                    if (p != rank) {
                        send_distance_update(vertex_id, new_dist, thread_id);
                    }
                }
            }
        }
        updates_to_send.clear();
        
        // Wait for all pending communications to complete
        if (!pending_requests.empty()) {
            MPI_Waitall(pending_requests.size(), pending_requests.data(), MPI_STATUSES_IGNORE);
            pending_requests.clear();
        }
        
        // Process any incoming messages
        process_incoming_messages();
        
        // Algorithm 3: Update Affected Vertices in parallel
        // First handle vertices affected by deletions
        bool global_has_affected_del = true;
        while (global_has_affected_del) {
            bool local_has_affected_del = false;
            
            // Check if there are any locally affected vertices
            #pragma omp parallel for reduction(||:local_has_affected_del)
            for (int i = 0; i < affected_del.size(); i++) {
                auto it = affected_del.begin();
                advance(it, i);
                if (it->second) {
                    local_has_affected_del = true;
                }
            }
            
            // Check if any process has affected vertices
            MPI_Allreduce(&local_has_affected_del, &global_has_affected_del, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            
            if (!global_has_affected_del) break;
            
            updates_to_send.clear();
            
            // Process each locally affected vertex
            #pragma omp parallel
            {
                vector<pair<int, float>> local_updates;
                
                #pragma omp for nowait
                for (int i = 0; i < affected_del.size(); i++) {
                    auto it = affected_del.begin();
                    advance(it, i);
                    int vertex_id = it->first;
                    bool is_affected = it->second;
                    
                    if (is_affected) {
                        #pragma omp critical
                        {
                            affected_del[vertex_id] = false; // Reset for next iteration
                        }
                        
                        // Find all children of vertex_id in the SSSP tree (vertices where vertex_id is the predecessor)
                        for (const auto& [child_id, pred] : predecessor) {
                            if (pred == vertex_id) {
                                #pragma omp critical
                                {
                                    // Mark child as affected
                                    distance[child_id] = numeric_limits<float>::infinity();
                                    affected_del[child_id] = true;
                                    affected[child_id] = true;
                                    
                                    // Add to updates to send
                                    local_updates.push_back({child_id, numeric_limits<float>::infinity()});
                                }
                            }
                        }
                    }
                }
                
                #pragma omp critical
                {
                    updates_to_send.insert(updates_to_send.end(), local_updates.begin(), local_updates.end());
                }
            }
            
            // Send updates to other processes
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                #pragma omp for
                for (size_t i = 0; i < updates_to_send.size(); i++) {
                    int vertex_id = updates_to_send[i].first;
                    float new_dist = updates_to_send[i].second;
                    
                    // For each process that might need this update
                    for (int p = 0; p < world_size; ++p) {
                        if (p != rank) {
                            send_distance_update(vertex_id, new_dist, thread_id);
                        }
                    }
                }
            }
            updates_to_send.clear();
            
            // Wait for all pending communications to complete
            if (!pending_requests.empty()) {
                MPI_Waitall(pending_requests.size(), pending_requests.data(), MPI_STATUSES_IGNORE);
                pending_requests.clear();
            }
            
            // Process any incoming messages
            process_incoming_messages();
        }
        
        // Now handle all affected vertices
        bool global_has_affected = true;
        while (global_has_affected) {
            bool local_has_affected = false;
            
            // Check if there are any locally affected vertices
            #pragma omp parallel for reduction(||:local_has_affected)
            for (int i = 0; i < affected.size(); i++) {
                auto it = affected.begin();
                advance(it, i);
                if (it->second) {
                    local_has_affected = true;
                }
            }
            
            // Check if any process has affected vertices
            MPI_Allreduce(&local_has_affected, &global_has_affected, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            
            if (!global_has_affected) break;
            
            updates_to_send.clear();
            
            // Process each locally affected vertex
            #pragma omp parallel
            {
                vector<pair<int, float>> local_updates;
                
                #pragma omp for nowait
                for (int i = 0; i < affected.size(); i++) {
                    auto it = affected.begin();
                    advance(it, i);
                    int vertex_id = it->first;
                    bool is_affected = it->second;
                    
                    if (is_affected) {
                        #pragma omp critical
                        {
                            affected[vertex_id] = false; // Reset for next iteration
                        }
                        
                        // Find the vertex in our local list
                        Vertex* vertex_ptr = nullptr;
                        for (auto& v : vertices) {
                            if (v.id == vertex_id) {
                                vertex_ptr = &v;
                                break;
                            }
                        }
                        
                        // Skip if we don't have this vertex locally
                        if (!vertex_ptr) continue;
                        
                        // Check all neighbors of vertex_id
                        for (size_t i = 0; i < vertex_ptr->neighbors.size(); ++i) {
                            int neighbor_id = vertex_ptr->neighbors[i].id;
                            float weight = vertex_ptr->weights[i];
                            
                            // Check if we can improve the path to neighbor through vertex_id
                            if (distance[neighbor_id] > distance[vertex_id] + weight) {
                                #pragma omp critical
                                {
                                    distance[neighbor_id] = distance[vertex_id] + weight;
                                    predecessor[neighbor_id] = vertex_id;
                                    affected[neighbor_id] = true;
                                    
                                    // Add to updates to send
                                    local_updates.push_back({neighbor_id, distance[neighbor_id]});
                                }
                            }
                            // Or vice versa
                            else if (distance[vertex_id] > distance[neighbor_id] + weight) {
                                #pragma omp critical
                                {
                                    distance[vertex_id] = distance[neighbor_id] + weight;
                                    predecessor[vertex_id] = neighbor_id;
                                    affected[vertex_id] = true;
                                    
                                    // Add to updates to send
                                    local_updates.push_back({vertex_id, distance[vertex_id]});
                                }
                            }
                        }
                    }
                }
                
                #pragma omp critical
                {
                    updates_to_send.insert(updates_to_send.end(), local_updates.begin(), local_updates.end());
                }
            }
            
            // Send updates to other processes
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                #pragma omp for
                for (size_t i = 0; i < updates_to_send.size(); i++) {
                    int vertex_id = updates_to_send[i].first;
                    float new_dist = updates_to_send[i].second;
                    
                    // For each process that might need this update
                    for (int p = 0; p < world_size; ++p) {
                        if (p != rank) {
                            send_distance_update(vertex_id, new_dist, thread_id);
                        }
                    }
                }
            }
            updates_to_send.clear();
            
            // Wait for all pending communications to complete
            if (!pending_requests.empty()) {
                MPI_Waitall(pending_requests.size(), pending_requests.data(), MPI_STATUSES_IGNORE);
                pending_requests.clear();
            }
            
            // Process any incoming messages
            process_incoming_messages();
        }
        
        // Ensure source vertex has distance 0 and no predecessor
        auto source_it = vertex_to_partition.begin();
        while (source_it != vertex_to_partition.end()) {
            if (distance[source_it->first] == 0.0f) {
                predecessor[source_it->first] = -1;
                break;
            }
            ++source_it;
        }
        
        // Final global synchronization to ensure all processes have consistent state
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

map<int, int> buildVertexPartition(vector<string>& lines) {
    map<int, int> vertexPartition;
    omp_lock_t mapLock;
    omp_init_lock(&mapLock);  // Initialize the lock

    #pragma omp parallel for
    for (size_t i = 0; i < lines.size(); i++) {
        istringstream iss(lines[i]);
        int partition, vertexId;
        
        if (!(iss >> partition >> vertexId)) {
            #pragma omp critical
            {
                cerr << "Error parsing line: " << lines[i] << endl;
            }
            continue;
        }

        // Only lock when modifying the shared map
        omp_set_lock(&mapLock);
        vertexPartition[vertexId] = partition;
        omp_unset_lock(&mapLock);
    }

    omp_destroy_lock(&mapLock);  // Clean up the lock
    return vertexPartition;
}

vector<Vertex> buildListOfVertices(vector<string>& lines, map<int, int>& vertexPartition, int rank) {
    vector<Vertex> listOfVertices;
    vector<int> missingPartitionInfo;
    omp_lock_t missingLock;
    omp_init_lock(&missingLock);

    // First pass - count vertices for this rank to pre-allocate
    int vertexCount = 0;
    #pragma omp parallel for reduction(+:vertexCount)
    for (size_t i = 0; i < lines.size(); i++) {
        istringstream iss(lines[i]);
        int partition, vertexId;
        if (iss >> partition >> vertexId && partition == rank) {
            vertexCount++;
        }
    }
    listOfVertices.reserve(vertexCount);

    // Second pass - process lines in parallel
    #pragma omp parallel
    {
        vector<Vertex> localVertices;
        vector<int> localMissing;
        
        #pragma omp for nowait
        for (size_t i = 0; i < lines.size(); i++) {
            const auto& line = lines[i];
            istringstream iss(line);
            int partition, vertexId;
            
            if (!(iss >> partition >> vertexId)) continue;
            if (partition != rank) continue;

            Vertex vertex;
            vertex.id = vertexId;
            vertex.partitionId = partition;

            int neighborId;
            while (iss >> neighborId) {
                Vertex neighbor;
                neighbor.id = neighborId;
                
                // Check partition (thread-safe with critical section)
                #pragma omp critical(partition_lookup)
                {
                    auto it = vertexPartition.find(neighborId);
                    neighbor.partitionId = (it != vertexPartition.end()) ? it->second : rank;
                    if (it == vertexPartition.end()) {
                        localMissing.push_back(neighborId);
                    }
                }
                
                vertex.weights.push_back(1.0f);
                vertex.neighbors.push_back(neighbor);
            }

            localVertices.push_back(vertex);
        }

        // Merge thread-local results
        #pragma omp critical(merge_results)
        {
            listOfVertices.insert(listOfVertices.end(), 
                                  localVertices.begin(), 
                                  localVertices.end());
            missingPartitionInfo.insert(missingPartitionInfo.end(),
                                      localMissing.begin(),
                                      localMissing.end());
        }
    }

    // Report missing partitions
    if (!missingPartitionInfo.empty()) {
        sort(missingPartitionInfo.begin(), missingPartitionInfo.end());
        missingPartitionInfo.erase(unique(missingPartitionInfo.begin(), 
                                        missingPartitionInfo.end()), 
                                 missingPartitionInfo.end());
        
        cerr << "Process " << rank << " found " << missingPartitionInfo.size() 
             << " neighbors with unknown partition assignment." << endl;
        
        for (size_t i = 0; i < min(5ul, missingPartitionInfo.size()); i++) {
            cerr << "  Missing partition for vertex ID: " << missingPartitionInfo[i] << endl;
        }
        if (missingPartitionInfo.size() > 5) {
            cerr << "  ... and " << (missingPartitionInfo.size() - 5) << " more" << endl;
        }
    }

    omp_destroy_lock(&missingLock);
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

// Constants for MPI message tags
const int INSERT_EDGE_TAG = 100;
const int DELETE_EDGE_TAG = 200;
const int UPDATE_COMPLETE_TAG = 300;
const int EXIT_LOOP_TAG = 400;

// Structure for sending edge information
struct EdgeMessage {
    int u;      // First vertex
    int v;      // Second vertex
    float weight; // Edge weight (default to 1.0 if not specified)
};

// Helper function to add an edge to a vertex in the local vertex list
void addEdgeToVertex(vector<Vertex>& vertices, int source_id, int target_id, float weight, 
                    const map<int, int>& vertexPartitions, int myRank) {
    // First, handle the source vertex which we know belongs to this process
    bool source_found = false;
    
    #pragma omp parallel for shared(source_found, vertices)
    for (size_t i = 0; i < vertices.size(); i++) {
        // Skip if source already found by another thread
        if (source_found) continue;
        
        if (vertices[i].id == source_id) {
            #pragma omp critical
            {
                if (!source_found) { // Double-check inside critical section
                    // Check if edge already exists
                    bool edge_exists = false;
                    for (size_t j = 0; j < vertices[i].neighbors.size(); j++) {
                        if (vertices[i].neighbors[j].id == target_id) {
                            // Edge already exists, update weight
                            vertices[i].weights[j] = weight;
                            edge_exists = true;
                            break;
                        }
                    }
                    
                    // If edge doesn't exist, add it
                    if (!edge_exists) {
                        // Create a temporary vertex for the neighbor
                        Vertex target_vertex;
                        target_vertex.id = target_id;
                        
                        // Find the partition ID for this vertex if known
                        if (vertexPartitions.find(target_id) != vertexPartitions.end()) {
                            target_vertex.partitionId = vertexPartitions.at(target_id);
                        } else {
                            target_vertex.partitionId = myRank;
                        }
                        
                        vertices[i].neighbors.push_back(target_vertex);
                        vertices[i].weights.push_back(weight);
                    }
                    
                    source_found = true;
                }
            }
        }
    }
    
    if (!source_found) {
        cerr << "Error: Source vertex " << source_id << " not found in process " << myRank << endl;
        return;
    }
    
    // Now handle the target vertex
    // First check if we have the target vertex locally
    bool target_found = false;
    
    #pragma omp parallel for shared(target_found, vertices)
    for (size_t i = 0; i < vertices.size(); i++) {
        // Skip if target already found by another thread
        if (target_found) continue;
        
        if (vertices[i].id == target_id) {
            #pragma omp critical
            {
                if (!target_found) { // Double-check inside critical section
                    // Check if edge already exists in reverse direction
                    bool edge_exists = false;
                    for (size_t j = 0; j < vertices[i].neighbors.size(); j++) {
                        if (vertices[i].neighbors[j].id == source_id) {
                            // Edge already exists, update weight
                            vertices[i].weights[j] = weight;
                            edge_exists = true;
                            break;
                        }
                    }
                    
                    // If edge doesn't exist, add it
                    if (!edge_exists) {
                        // Create a temporary vertex for the neighbor
                        Vertex source_vertex;
                        source_vertex.id = source_id;
                        source_vertex.partitionId = myRank; // We know this vertex belongs to current process
                        
                        vertices[i].neighbors.push_back(source_vertex);
                        vertices[i].weights.push_back(weight);
                    }
                    
                    target_found = true;
                }
            }
        }
    }
    
    // If target vertex is not local, send MPI message to the appropriate process
    if (!target_found) {
        // Find which process owns target vertex
        if (vertexPartitions.find(target_id) != vertexPartitions.end()) {
            int target_process = vertexPartitions.at(target_id);
            
            // Don't send to ourselves
            if (target_process != myRank) {
                // Create and send a message to add source_id as neighbor to target_id
                EdgeMessage msg = {target_id, source_id, weight}; // Note: reversed order
                MPI_Send(&msg, sizeof(EdgeMessage), MPI_BYTE, target_process, INSERT_EDGE_TAG, MPI_COMM_WORLD);
                
                cout << "Rank " << myRank << ": Sent request to process " << target_process 
                     << " to add " << source_id << " as neighbor to " << target_id << endl;
            }
        } else {
            cerr << "Warning: Target vertex " << target_id << " not found in partition map" << endl;
        }
    }
}

// Helper function to remove an edge from a vertex in the local vertex list
void removeEdgeFromVertex(vector<Vertex>& vertices, int source_id, int target_id, 
                         const map<int, int>& vertexPartitions, int myRank) {
    // First, handle the source vertex which we know belongs to this process
    bool source_found = false;
    
    #pragma omp parallel for shared(source_found, vertices)
    for (size_t i = 0; i < vertices.size(); i++) {
        // Skip if source already found by another thread
        if (source_found) continue;
        
        if (vertices[i].id == source_id) {
            #pragma omp critical
            {
                if (!source_found) { // Double-check inside critical section
                    // Find and remove the edge from source to target
                    for (size_t j = 0; j < vertices[i].neighbors.size(); j++) {
                        if (vertices[i].neighbors[j].id == target_id) {
                            // Found the neighbor to remove
                            vertices[i].neighbors.erase(vertices[i].neighbors.begin() + j);
                            vertices[i].weights.erase(vertices[i].weights.begin() + j);
                            break;
                        }
                    }
                    
                    source_found = true;
                }
            }
        }
    }
    
    if (!source_found) {
        cerr << "Error: Source vertex " << source_id << " not found in process " << myRank << endl;
        return;
    }
    
    // Now handle the target vertex
    // First check if we have the target vertex locally
    bool target_found = false;
    
    #pragma omp parallel for shared(target_found, vertices)
    for (size_t i = 0; i < vertices.size(); i++) {
        // Skip if target already found by another thread
        if (target_found) continue;
        
        if (vertices[i].id == target_id) {
            #pragma omp critical
            {
                if (!target_found) { // Double-check inside critical section
                    // Find and remove the edge from target to source
                    for (size_t j = 0; j < vertices[i].neighbors.size(); j++) {
                        if (vertices[i].neighbors[j].id == source_id) {
                            // Found the neighbor to remove
                            vertices[i].neighbors.erase(vertices[i].neighbors.begin() + j);
                            vertices[i].weights.erase(vertices[i].weights.begin() + j);
                            break;
                        }
                    }
                    
                    target_found = true;
                }
            }
        }
    }
    
    // If target vertex is not local, send MPI message to the appropriate process
    if (!target_found) {
        // Find which process owns target vertex
        if (vertexPartitions.find(target_id) != vertexPartitions.end()) {
            int target_process = vertexPartitions.at(target_id);
            
            // Don't send to ourselves
            if (target_process != myRank) {
                // Create and send a message to remove source_id as neighbor from target_id
                EdgeMessage msg = {target_id, source_id, 0.0f}; // Note: reversed order, weight doesn't matter for deletion
                MPI_Send(&msg, sizeof(EdgeMessage), MPI_BYTE, target_process, DELETE_EDGE_TAG, MPI_COMM_WORLD);
                
                cout << "Rank " << myRank << ": Sent request to process " << target_process 
                     << " to remove " << source_id << " as neighbor from " << target_id << endl;
            }
        } else {
            cerr << "Warning: Target vertex " << target_id << " not found in partition map" << endl;
        }
    }
}

// Function to signal end of updates to all processes
void signalEndOfUpdates(int size) {
    #pragma omp parallel for
    for (int p = 1; p < size; p++) {
        int signal = 1;
        #pragma omp critical
        {
            MPI_Send(&signal, 1, MPI_INT, p, EXIT_LOOP_TAG, MPI_COMM_WORLD);
        }
    }
}

// Function to check for exit signal
bool checkForExitSignal() {
    MPI_Status status;
    int flag = 0;
    
    MPI_Iprobe(MPI_ANY_SOURCE, EXIT_LOOP_TAG, MPI_COMM_WORLD, &flag, &status);
    
    if (flag) {
        int signal;
        MPI_Recv(&signal, 1, MPI_INT, status.MPI_SOURCE, EXIT_LOOP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return true;
    }
    
    return false;
}

// Function to process edge insertions
bool processEdgeInsertions(vector<Vertex>& vertices, vector<pair<int, int>>& local_insertions, 
                          const map<int, int>& vertexPartitions, int rank) {
    MPI_Status status;
    int flag = 0;
    
    MPI_Iprobe(MPI_ANY_SOURCE, INSERT_EDGE_TAG, MPI_COMM_WORLD, &flag, &status);
    
    if (flag) {
        EdgeMessage msg;
        MPI_Recv(&msg, sizeof(EdgeMessage), MPI_BYTE, status.MPI_SOURCE, INSERT_EDGE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Process the edge insertion locally
        addEdgeToVertex(vertices, msg.u, msg.v, msg.weight, vertexPartitions, rank);
        
        #pragma omp critical
        local_insertions.push_back({msg.u, msg.v});
        
        cout << "Rank " << rank << ": Received and added edge (" << msg.u << "," << msg.v << ")" << endl;
        return true;
    }
    
    return false;
}

// Function to process edge deletions
bool processEdgeDeletions(vector<Vertex>& vertices, vector<pair<int, int>>& local_deletions, 
                         const map<int, int>& vertexPartitions, int rank) {
    MPI_Status status;
    int flag = 0;
    
    MPI_Iprobe(MPI_ANY_SOURCE, DELETE_EDGE_TAG, MPI_COMM_WORLD, &flag, &status);
    
    if (flag) {
        EdgeMessage msg;
        MPI_Recv(&msg, sizeof(EdgeMessage), MPI_BYTE, status.MPI_SOURCE, DELETE_EDGE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Process the edge deletion locally
        removeEdgeFromVertex(vertices, msg.u, msg.v, vertexPartitions, rank);
        
        #pragma omp critical
        local_deletions.push_back({msg.u, msg.v});
        
        cout << "Rank " << rank << ": Received and removed edge (" << msg.u << "," << msg.v << ")" << endl;
        return true;
    }
    
    return false;
}

// Function to send edge insertions from coordinator to appropriate processes
void sendEdgeInsertions(const vector<pair<int, int>>& insertions, vector<Vertex>& vertices, 
                       vector<pair<int, int>>& local_insertions, const map<int, int>& vertexPartitions, int rank, int size) {
    #pragma omp parallel
    {
        vector<pair<int, int>> thread_local_insertions;
        
        #pragma omp for schedule(dynamic, NUM_THREADS)
        for (size_t i = 0; i < insertions.size(); i++) {
            int u = insertions[i].first;
            int v = insertions[i].second;
            
            // Find which process owns vertex u
            if (vertexPartitions.find(u) == vertexPartitions.end()) {
                #pragma omp critical
                {
                    cerr << "Warning: Vertex " << u << " not found in partition map" << endl;
                }
                continue;
            }
            
            int target_process = vertexPartitions.at(u);

            if (target_process == rank) {
                addEdgeToVertex(vertices, u, v, 1.0f, vertexPartitions, rank); // Using default weight of 1.0
                thread_local_insertions.push_back({u, v});
                
                #pragma omp critical
                {
                    cout << "Rank " << rank << ": Added edge (" << u << "," << v << ") locally" << endl;
                }
                continue;
            }
            
            // Create and send the message
            EdgeMessage msg = {u, v, 1.0f}; // Using default weight of 1.0
            
            #pragma omp critical
            {
                MPI_Send(&msg, sizeof(EdgeMessage), MPI_BYTE, target_process, INSERT_EDGE_TAG, MPI_COMM_WORLD);
                cout << "Rank " << rank << ": Sent insertion request for edge (" << u << "," << v << ") to process " << target_process << endl;
            }
        }
        
        // Merge thread-local results into the shared vector
        #pragma omp critical
        {
            local_insertions.insert(local_insertions.end(), thread_local_insertions.begin(), thread_local_insertions.end());
        }
    }
}

// Function to send edge deletions from coordinator to appropriate processes
void sendEdgeDeletions(const vector<pair<int, int>>& deletions, vector<Vertex>& vertices, 
                      vector<pair<int, int>>& local_deletions, const map<int, int>& vertexPartitions, int rank, int size) {
    #pragma omp parallel
    {
        vector<pair<int, int>> thread_local_deletions;
        
        #pragma omp for schedule(dynamic, NUM_THREADS)
        for (size_t i = 0; i < deletions.size(); i++) {
            int u = deletions[i].first;
            int v = deletions[i].second;
            
            // Find which process owns vertex u
            if (vertexPartitions.find(u) == vertexPartitions.end()) {
                #pragma omp critical
                {
                    cerr << "Warning: Vertex " << u << " not found in partition map" << endl;
                }
                continue;
            }
            
            int target_process = vertexPartitions.at(u);

            if (target_process == rank) {
                removeEdgeFromVertex(vertices, u, v, vertexPartitions, rank);
                thread_local_deletions.push_back({u, v});
                
                #pragma omp critical
                {
                    cout << "Rank " << rank << ": Removed edge (" << u << "," << v << ") locally" << endl;
                }
                continue;
            }
            
            // Create and send the message
            EdgeMessage msg = {u, v, 0.0f}; // Weight doesn't matter for deletion
            
            #pragma omp critical
            {
                MPI_Send(&msg, sizeof(EdgeMessage), MPI_BYTE, target_process, DELETE_EDGE_TAG, MPI_COMM_WORLD);
                cout << "Rank " << rank << ": Sent deletion request for edge (" << u << "," << v << ") to process " << target_process << endl;
            }
        }
        
        // Merge thread-local results into the shared vector
        #pragma omp critical
        {
            local_deletions.insert(local_deletions.end(), thread_local_deletions.begin(), thread_local_deletions.end());
        }
    }
}

// Function to receive and process all edge update messages
void receiveAndProcessEdgeUpdates(vector<Vertex>& vertices, vector<pair<int, int>>& local_insertions, 
                                 vector<pair<int, int>>& local_deletions, const map<int, int>& vertexPartitions, int rank) {
    bool processing_updates = true;
    
    // Create an array to track thread-specific processing flags
    int max_threads = omp_get_max_threads();
    vector<bool> thread_processed(max_threads, false);
    
    while (processing_updates) {
        bool processed_message = false;
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            thread_processed[thread_id] = false;
            
            // We need to be careful with message processing in parallel
            // Using a strategy where threads take turns checking for messages
            #pragma omp for ordered
            for (int i = 0; i < max_threads; i++) {
                #pragma omp ordered
                {
                    // Check for insertion messages
                    if (processEdgeInsertions(vertices, local_insertions, vertexPartitions, rank)) {
                        thread_processed[thread_id] = true;
                    }
                    
                    // Check for deletion messages
                    if (processEdgeDeletions(vertices, local_deletions, vertexPartitions, rank)) {
                        thread_processed[thread_id] = true;
                    }
                }
            }
        }
        
        // Check if any thread processed a message
        for (int i = 0; i < max_threads; i++) {
            processed_message |= thread_processed[i];
        }
        
        // Check for exit loop signal - this cannot be parallelized due to MPI_Iprobe/MPI_Recv
        if (checkForExitSignal()) {
            processing_updates = false;
            processed_message = true;
        }
    }
}

// Function to perform dynamic graph updates with timing
void performDynamicGraphUpdates(ParallelDijkstra& dijkstra, vector<Vertex>& vertices, 
                               map<int, int>& vertexPartitions, int rank, int size) {
    // Predefined insertions and deletions (similar to the serial version)
    vector<pair<int, int>> all_insertions = {
        {6, 11},    // Connects mid-level nodes (original 5-10)
        {9, 17},    // Creates a shortcut (original 8-16)
        {4, 8},     // Cross-branch link (original 3-7)
        {1, 11},    // Root-to-mid-level jump (original 0-10)
        {13, 25}    // Deep-level connection (original 12-24)
    };
    
    vector<pair<int, int>> all_deletions = {
        {1, 6},     // Root-level edge removal (original 0-5)
        {11, 15},   // Chain breaker (original 10-14)
        {3, 8},     // Alternative path removal (original 2-7)
        {7, 11},    // Mid-level connection (original 6-10)
        {17, 21}    // Deep-level removal (original 16-20)
    };

    // Local collections to track updates on this process
    vector<pair<int, int>> local_insertions;
    vector<pair<int, int>> local_deletions;
    
    // Synchronize all processes before starting timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== BEGIN TIMING FOR TOTAL UPDATE PROCESS =====
    auto total_start = chrono::high_resolution_clock::now();
    
    if (rank == 0) {
        // Process 0 coordinates the dynamic updates
        cout << "Rank 0: Starting dynamic updates" << endl;
        
        // Send edge insertions
        sendEdgeInsertions(all_insertions, vertices, local_insertions, vertexPartitions, rank, size);
        
        // Send edge deletions
        sendEdgeDeletions(all_deletions, vertices, local_deletions, vertexPartitions, rank, size);
        
        // Signal end of updates
        signalEndOfUpdates(size);
    }
    
    // All processes (excluding rank 0) receive and process messages
    if (rank != 0)
        receiveAndProcessEdgeUpdates(vertices, local_insertions, local_deletions, vertexPartitions, rank);
    
    // Synchronize all processes after edge modifications before starting SSSP update
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== END TIMING FOR EDGE MODIFICATIONS =====
    auto edge_end = chrono::high_resolution_clock::now();
    
    // ===== BEGIN TIMING FOR SSSP UPDATE =====
    auto sssp_start = chrono::high_resolution_clock::now();
    
    // After processing all edge updates, run the SSSP update algorithm
    cout << "Rank " << rank << ": Updating SSSP with " << local_insertions.size() 
         << " insertions and " << local_deletions.size() << " deletions" << endl;
         
    // Call the updateSSSP function with our collected changes
    dijkstra.updateSSSP(local_insertions, local_deletions);
    
    // ===== END TIMING FOR SSSP UPDATE =====
    auto sssp_end = chrono::high_resolution_clock::now();
    
    // Synchronize all processes after SSSP update
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== END TIMING FOR TOTAL UPDATE PROCESS =====
    auto total_end = chrono::high_resolution_clock::now();
    
    // Calculate durations in milliseconds
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count();
    auto edge_duration = chrono::duration_cast<chrono::milliseconds>(edge_end - total_start).count();
    auto sssp_duration = chrono::duration_cast<chrono::milliseconds>(sssp_end - sssp_start).count();
    
    // Local timing results
    cout << "\nPerformance Metrics for Rank " << rank << ":\n";
    cout << "Total update time: " << total_duration << " ms\n";
    cout << "Edge modification time: " << edge_duration << " ms\n";
    cout << "SSSP update time: " << sssp_duration << " ms\n";
    
    // Collect timing statistics from all processes
    long max_total_time = 0;
    long max_edge_time = 0;
    long max_sssp_time = 0;
    long sum_total_time = 0;
    long sum_edge_time = 0;
    long sum_sssp_time = 0;
    
    // Use MPI_Reduce to get the maximum and sum of each timing metric
    MPI_Reduce(&total_duration, &max_total_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&edge_duration, &max_edge_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sssp_duration, &max_sssp_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    MPI_Reduce(&total_duration, &sum_total_time, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&edge_duration, &sum_edge_time, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sssp_duration, &sum_sssp_time, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Only rank 0 reports the global timing results
    if (rank == 0) {
        long avg_total_time = sum_total_time / size;
        long avg_edge_time = sum_edge_time / size;
        long avg_sssp_time = sum_sssp_time / size;
        
        cout << "\n===== GLOBAL PERFORMANCE METRICS =====\n";
        cout << "Maximum total update time across all processes: " << max_total_time << " ms\n";
        cout << "Maximum edge modification time across all processes: " << max_edge_time << " ms\n";
        cout << "Maximum SSSP update time across all processes: " << max_sssp_time << " ms\n";
        
        cout << "\nAverage total update time across all processes: " << avg_total_time << " ms\n";
        cout << "Average edge modification time across all processes: " << avg_edge_time << " ms\n";
        cout << "Average SSSP update time across all processes: " << avg_sssp_time << " ms\n";
    }
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
        graph.loadGraphFromFile("../../graphs/p2p-Gnutella-small.txt");
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

    MPI_Barrier(MPI_COMM_WORLD);
    
    // -------- DYNAMIC UPDATES SECTION --------
    performDynamicGraphUpdates(dijkstra, listOfVertices, vertexPartitions, rank, size);
    // -------- END DYNAMIC UPDATES SECTION --------

    MPI_Barrier(MPI_COMM_WORLD);

    // Print final results
    const auto& distances = dijkstra.get_distances();
    const auto& predecessors = dijkstra.get_predecessors();

    // Print local results
    // for (const auto& vertex : listOfVertices) {
    //     cout << "Rank " << rank << " - Vertex " << vertex.id << ": distance = " << distances.at(vertex.id);
    //     if (predecessors.find(vertex.id) != predecessors.end()) {
    //         cout << ", predecessor = " << predecessors.at(vertex.id);
    //     }
    //     cout << endl;
    // }
    
    MPI_Finalize();
    return 0;
}