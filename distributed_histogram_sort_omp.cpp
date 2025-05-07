#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>   // for std::iota
#include <omp.h>

// Total number of elements to sort
const int DATA_SIZE   = 1'000'000;
// Number of threads / bins
const int NUM_THREADS = 8;
const int NUM_BINS    = NUM_THREADS;
// (MAX_VAL is no longer needed for generation)

void generate_data(std::vector<int>& data) {
    // 1) Fill with unique values 1..DATA_SIZE
    std::iota(data.begin(), data.end(), 1);

    // 2) Shuffle
    std::mt19937 rng(42);
    std::shuffle(data.begin(), data.end(), rng);
}

void distributed_histogram_sort(std::vector<int>& data) {
    int n = data.size();

    // Now our range is [1..DATA_SIZE], so:
    int max_val  = DATA_SIZE;
    int bin_size = (max_val + NUM_BINS - 1) / NUM_BINS;

    // 1) Prepare global bins and per‑bin locks
    std::vector<std::vector<int>> bins(NUM_BINS);
    std::vector<omp_lock_t> locks(NUM_BINS);
    for (int b = 0; b < NUM_BINS; ++b) omp_init_lock(&locks[b]);

    // 2) Each thread bins its chunk locally, then merges under locks
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        int chunk = (n + NUM_THREADS - 1) / NUM_THREADS;
        int start = tid * chunk;
        int end   = std::min(n, start + chunk);

        std::vector<std::vector<int>> local_bins(NUM_BINS);
        for (int i = start; i < end; ++i) {
            int b = (data[i] - 1) / bin_size;  // shift to 0‑based
            if (b >= NUM_BINS) b = NUM_BINS - 1;
            local_bins[b].push_back(data[i]);
        }

        // merge local -> global
        for (int b = 0; b < NUM_BINS; ++b) {
            omp_set_lock(&locks[b]);
            bins[b].insert(bins[b].end(),
                           local_bins[b].begin(),
                           local_bins[b].end());
            omp_unset_lock(&locks[b]);
        }
    }
    for (int b = 0; b < NUM_BINS; ++b) omp_destroy_lock(&locks[b]);

    // 3) Sort each bin in parallel
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
    for (int b = 0; b < NUM_BINS; ++b) {
        std::sort(bins[b].begin(), bins[b].end());
    }

    // 4) Concatenate back
    int idx = 0;
    for (int b = 0; b < NUM_BINS; ++b) {
        for (int x : bins[b]) {
            data[idx++] = x;
        }
    }
}

int main() {
    std::vector<int> data(DATA_SIZE);
    generate_data(data);

    std::cout << "First 10 before sort: ";
    for (int i = 0; i < 10; ++i) std::cout << data[i] << " ";
    std::cout << "\n";

    double t0 = omp_get_wtime();
    distributed_histogram_sort(data);
    double t1 = omp_get_wtime();

    std::cout << "First 10 after  sort: ";
    for (int i = 0; i < 10; ++i) std::cout << data[i] << " ";
    std::cout << "\n";

    std::cout << "Elapsed time: " << (t1 - t0) << " seconds\n";
    return 0;
}
