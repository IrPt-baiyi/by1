/* rng-probe.cpp -- T25-2 judge: same-seed RNG parity oracle.
 *
 * Prints, for a seed and a probability vector:
 *   raw-1: <16 raw mt19937 uint32 draws>
 *   canon: <8 generate_canonical<double,53> values>
 *   idx:   <N std::discrete_distribution draws>
 *
 * by1sample.sample_mt must reproduce all three (mt19937 core + libstdc++
 * generate_canonical mapping + discrete_distribution lower_bound). Same
 * compiler = same libstdc++, so this binary IS the oracle.
 *
 * Build: g++ -O2 -o tools/rng-probe.exe tools/rng-probe.cpp
 * Usage: rng-probe <seed> <n_idx> <p1,p2,p3,...>
 */
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: rng-probe <seed> <n_idx> <p1,p2,...>\n");
        return 2;
    }
    unsigned seed = (unsigned) atoi(argv[1]);
    int n_idx = atoi(argv[2]);
    std::vector<float> probs;
    std::string ps = argv[3];
    size_t pos = 0;
    while (pos < ps.size()) {
        size_t comma = ps.find(',', pos);
        if (comma == std::string::npos) {
            comma = ps.size();
        }
        probs.push_back((float) atof(ps.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
    }

    std::mt19937 rng_raw(seed);
    printf("raw-1:");
    for (int i = 0; i < 16; ++i) {
        printf(" %u", rng_raw());
    }
    printf("\n");

    std::mt19937 rng_canon(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    printf("canon:");
    for (int i = 0; i < 8; ++i) {
        printf(" %.17g", u(rng_canon));
    }
    printf("\n");

    std::mt19937 rng(seed);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    printf("idx:");
    for (int i = 0; i < n_idx; ++i) {
        printf(" %d", dist(rng));
    }
    printf("\n");
    return 0;
}
