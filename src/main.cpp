#include <iostream>
#include <vector>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total processes

    const int array_size = 100;  // Total elements
    std::vector<int> global_array;

    // Root process initializes the full array
    if (rank == 0) {
        global_array.resize(array_size);
        for (int i = 0; i < array_size; ++i) {
            global_array[i] = i + 1;  // Fill with 1, 2, ..., 100
        }
    }

    // Determine chunk size per process
    int chunk_size = array_size / size;
    std::vector<int> local_array(chunk_size);

    // Scatter the array to all processes
    MPI_Scatter(
        global_array.data(), chunk_size, MPI_INT,  // Send buffer
        local_array.data(), chunk_size, MPI_INT,   // Receive buffer
        0, MPI_COMM_WORLD                         // Root process
    );

    // Compute local sum
    int local_sum = 0;
    for (int num : local_array) {
        local_sum += num;
    }
    std::cout << "Process " << rank << " local sum: " << local_sum << std::endl;

    // Gather all partial sums at root
    std::vector<int> all_sums(size);
    MPI_Gather(
        &local_sum, 1, MPI_INT,        // Send local sum
        all_sums.data(), 1, MPI_INT,   // Receive all sums
        0, MPI_COMM_WORLD              // Root process
    );

    // Root computes the final sum
    if (rank == 0) {
        int total_sum = 0;
        for (int sum : all_sums) {
            total_sum += sum;
        }
        std::cout << "Total sum: " << total_sum << std::endl;
    }

    MPI_Finalize();  // Clean up MPI
    return 0;
}