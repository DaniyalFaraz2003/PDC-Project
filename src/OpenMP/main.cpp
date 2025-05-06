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
    
        #pragma omp parallel for reduction(+:numEdges) schedule(dynamic, 1024)
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
            #pragma omp parallel for schedule(dynamic, 16)
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
        for (const auto& vertex : vertices) {
            int vertex_id = vertex.id;
            #pragma omp critical
            {
                affected[vertex_id] = false;
                affected_del[vertex_id] = false;
            }
            
            // Also initialize for neighbors we know about
            for (const auto& neighbor : vertex.neighbors) {
                #pragma omp critical
                {
                    affected[neighbor.id] = false;
                    affected_del[neighbor.id] = false;
                }
            }
        }
        
        // Make sure all known vertices from distance map are included
        #pragma omp parallel for
        for (auto it = distance.begin(); it != distance.end(); ++it) {
            int vertex_id = it->first;
            #pragma omp critical
            {
                if (affected.find(vertex_id) == affected.end()) {
                    affected[vertex_id] = false;
                    affected_del[vertex_id] = false;
                }
            }
        }
        
        // Thread-safe container for updates that need to be sent to other processes
        vector<pair<int, float>> updates_to_send;
        mutex updates_mutex;
        
        // Process deleted edges first
        #pragma omp parallel for
        for (size_t i = 0; i < deletions.size(); ++i) {
            const auto& edge = deletions[i];
            int u = edge.first;
            int v = edge.second;
            
            bool should_process = false;
            bool check_predecessor_v = false;
            bool check_predecessor_u = false;
            
            // Thread-safe check if we know about these vertices
            #pragma omp critical
            {
                should_process = (distance.find(u) != distance.end() && distance.find(v) != distance.end());
                if (should_process) {
                    check_predecessor_v = (predecessor.find(v) != predecessor.end());
                    check_predecessor_u = (predecessor.find(u) != predecessor.end());
                }
            }
            
            if (!should_process) continue;
            
            // Check if edge is in the shortest path tree (using predecessor)
            bool is_in_tree = false;
            int affected_vertex = -1;
            
            #pragma omp critical
            {
                is_in_tree = (check_predecessor_v && predecessor[v] == u) || 
                             (check_predecessor_u && predecessor[u] == v);
                
                if (is_in_tree) {
                    // Determine which vertex to mark as affected (the one with higher distance)
                    affected_vertex = (distance[u] > distance[v]) ? u : v;
                    
                    // Set distance to infinity and mark as affected
                    distance[affected_vertex] = numeric_limits<float>::infinity();
                    affected_del[affected_vertex] = true;
                    affected[affected_vertex] = true;
                }
            }
            
            if (is_in_tree) {
                // Check if this is a local vertex that needs update propagation
                auto it = vertex_to_partition.find(affected_vertex);
                if (it != vertex_to_partition.end() && it->second == rank) {
                    lock_guard<mutex> lock(updates_mutex);
                    updates_to_send.push_back({affected_vertex, numeric_limits<float>::infinity()});
                }
            }
        }
        
        // Process inserted edges with OpenMP
        #pragma omp parallel for
        for (size_t i = 0; i < insertions.size(); ++i) {
            const auto& edge = insertions[i];
            int u = edge.first;
            int v = edge.second;
            
            bool should_process = false;
            float dist_u = 0.0f, dist_v = 0.0f;
            
            // Thread-safe check if we know about these vertices
            #pragma omp critical
            {
                should_process = (distance.find(u) != distance.end() && distance.find(v) != distance.end());
                if (should_process) {
                    dist_u = distance[u];
                    dist_v = distance[v];
                }
            }
            
            if (!should_process) continue;
            
            // Find the vertex with lower distance (x) and higher distance (y)
            int x = (dist_u > dist_v) ? v : u;
            int y = (dist_u > dist_v) ? u : v;
            
            // Find the weight of the edge from x to y
            float weight = 0.0f;
            bool weight_found = false;
            
            // Try to find vertex x in our local vertices
            for (const auto& vertex : vertices) {
                if (vertex.id == x) {
                    // Look for neighbor y
                    for (size_t j = 0; j < vertex.neighbors.size(); ++j) {
                        if (vertex.neighbors[j].id == y) {
                            weight = vertex.weights[j];
                            weight_found = true;
                            break;
                        }
                    }
                    if (weight_found) break;
                }
            }
            
            // If weight is 0, we might not have found the edge locally
            if (!weight_found) {
                // For simplicity, we'll use MPI to gather the weight from other processes
                // This is a simplified approach - a more optimized version would batch these requests
                float local_weight = weight;
                
                // We need to synchronize threads before MPI communication
                #pragma omp barrier
                #pragma omp single
                {
                    MPI_Allreduce(&local_weight, &weight, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                }
                
                // If still 0, this edge might not exist in any process
                if (weight == 0.0f) continue;
            }
            
            bool update_needed = false;
            float x_dist = 0.0f, y_dist = 0.0f;
            
            #pragma omp critical
            {
                x_dist = distance[x];
                y_dist = distance[y];
                
                // Check if the new edge provides a shorter path
                if (y_dist > x_dist + weight) {
                    distance[y] = x_dist + weight;
                    predecessor[y] = x;
                    affected[y] = true;
                    update_needed = true;
                }
            }
            
            if (update_needed) {
                // If this is a local vertex, we'll need to propagate this change
                auto it = vertex_to_partition.find(y);
                if (it != vertex_to_partition.end() && it->second == rank) {
                    lock_guard<mutex> lock(updates_mutex);
                    updates_to_send.push_back({y, distance[y]});
                }
            }
        }
        
        // Send initial updates to other processes (outside of parallel region)
        for (const auto& [vertex_id, new_dist] : updates_to_send) {
            for (int p = 0; p < world_size; ++p) {
                if (p != rank) {
                    send_distance_update(vertex_id, new_dist);
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
            // Collect locally affected vertices
            vector<int> affected_vertices_del;
            
            #pragma omp parallel
            {
                vector<int> thread_affected;
                
                #pragma omp for
                for (auto it = affected_del.begin(); it != affected_del.end(); ++it) {
                    if (it->second) {
                        thread_affected.push_back(it->first);
                        it->second = false; // Reset for next iteration
                    }
                }
                
                #pragma omp critical
                {
                    affected_vertices_del.insert(affected_vertices_del.end(), 
                                              thread_affected.begin(), 
                                              thread_affected.end());
                }
            }
            
            bool local_has_affected_del = !affected_vertices_del.empty();
            
            // Check if any process has affected vertices
            MPI_Allreduce(&local_has_affected_del, &global_has_affected_del, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            
            if (!global_has_affected_del) break;
            
            // Process each locally affected vertex
            #pragma omp parallel
            {
                vector<pair<int, float>> thread_updates;
                
                #pragma omp for
                for (size_t i = 0; i < affected_vertices_del.size(); ++i) {
                    int vertex_id = affected_vertices_del[i];
                    
                    // Find all children of vertex_id in the SSSP tree
                    vector<int> affected_children;
                    
                    #pragma omp critical
                    {
                        for (const auto& [child_id, pred] : predecessor) {
                            if (pred == vertex_id) {
                                affected_children.push_back(child_id);
                            }
                        }
                    }
                    
                    // Process each affected child
                    for (int child_id : affected_children) {
                        #pragma omp critical
                        {
                            distance[child_id] = numeric_limits<float>::infinity();
                            affected_del[child_id] = true;
                            affected[child_id] = true;
                        }
                        
                        thread_updates.push_back({child_id, numeric_limits<float>::infinity()});
                    }
                }
                
                #pragma omp critical
                {
                    updates_to_send.insert(updates_to_send.end(), 
                                         thread_updates.begin(), 
                                         thread_updates.end());
                }
            }
            
            // Send updates to other processes (outside parallel region)
            for (const auto& [vertex_id, new_dist] : updates_to_send) {
                for (int p = 0; p < world_size; ++p) {
                    if (p != rank) {
                        send_distance_update(vertex_id, new_dist);
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
            // Collect locally affected vertices
            vector<int> affected_vertices;
            
            #pragma omp parallel
            {
                vector<int> thread_affected;
                
                #pragma omp for
                for (auto it = affected.begin(); it != affected.end(); ++it) {
                    if (it->second) {
                        thread_affected.push_back(it->first);
                        it->second = false; // Reset for next iteration
                    }
                }
                
                #pragma omp critical
                {
                    affected_vertices.insert(affected_vertices.end(), 
                                          thread_affected.begin(), 
                                          thread_affected.end());
                }
            }
            
            bool local_has_affected = !affected_vertices.empty();
            
            // Check if any process has affected vertices
            MPI_Allreduce(&local_has_affected, &global_has_affected, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            
            if (!global_has_affected) break;
            
            // Use OpenMP to process affected vertices in parallel
            #pragma omp parallel
            {
                vector<pair<int, float>> thread_updates;
                
                #pragma omp for schedule(dynamic)
                for (size_t i = 0; i < affected_vertices.size(); ++i) {
                    int vertex_id = affected_vertices[i];
                    
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
                    
                    float vertex_dist;
                    #pragma omp critical
                    {
                        vertex_dist = distance[vertex_id];
                    }
                    
                    // Process all neighbors
                    for (size_t j = 0; j < vertex_ptr->neighbors.size(); ++j) {
                        int neighbor_id = vertex_ptr->neighbors[j].id;
                        float weight = vertex_ptr->weights[j];
                        
                        float neighbor_dist;
                        bool update_needed = false;
                        bool update_type = false; // false for neighbor update, true for vertex update
                        
                        #pragma omp critical
                        {
                            neighbor_dist = distance[neighbor_id];
                            
                            // Check if we can improve the path to neighbor through vertex_id
                            if (neighbor_dist > vertex_dist + weight) {
                                distance[neighbor_id] = vertex_dist + weight;
                                predecessor[neighbor_id] = vertex_id;
                                affected[neighbor_id] = true;
                                update_needed = true;
                                update_type = false;
                            }
                            // Or vice versa
                            else if (vertex_dist > neighbor_dist + weight) {
                                distance[vertex_id] = neighbor_dist + weight;
                                predecessor[vertex_id] = neighbor_id;
                                affected[vertex_id] = true;
                                update_needed = true;
                                update_type = true;
                            }
                        }
                        
                        if (update_needed) {
                            int updated_id = update_type ? vertex_id : neighbor_id;
                            float updated_dist;
                            
                            #pragma omp critical
                            {
                                updated_dist = distance[updated_id];
                            }
                            
                            // Check if this is a vertex that needs update propagation
                            auto it = vertex_to_partition.find(updated_id);
                            if (it != vertex_to_partition.end() && it->second == rank) {
                                thread_updates.push_back({updated_id, updated_dist});
                            }
                        }
                    }
                }
                
                #pragma omp critical
                {
                    updates_to_send.insert(updates_to_send.end(), 
                                         thread_updates.begin(), 
                                         thread_updates.end());
                }
            }
            
            // Send updates to other processes (outside parallel region)
            for (const auto& [vertex_id, new_dist] : updates_to_send) {
                for (int p = 0; p < world_size; ++p) {
                    if (p != rank) {
                        send_distance_update(vertex_id, new_dist);
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
        int thread_count = omp_get_max_threads();
        vector<int> potential_sources(thread_count, -1);
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            
            #pragma omp for
            for (auto it = distance.begin(); it != distance.end(); ++it) {
                if (it->second == 0.0f) {
                    potential_sources[thread_id] = it->first;
                }
            }
        }
        
        for (int src : potential_sources) {
            if (src != -1) {
                predecessor[src] = -1;
                break;
            }
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
