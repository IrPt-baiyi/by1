/* sampler-probe.cpp -- T25-3 judge: chain-level token parity oracle.
 *
 * Builds the llama.cpp sampler chain [top_k, top_p, min_p, temp, dist(seed)]
 * (the same order by1sample.chain_f32 replicates), applies it to a logits
 * file (one float per line), and prints "seed token" for each seed in a
 * range. Same compiler = same libstdc++ = oracle.
 *
 * Build: g++ -O2 -o tools/sampler-probe.exe tools/sampler-probe.cpp
 *        -Illamacpp/llama.cpp/include -Illamacpp/llama.cpp/ggml/include
 *        llamacpp/build-cli/src/libllama.a llamacpp/build-cli/ggml/src/ggml.a
 *        llamacpp/build-cli/ggml/src/ggml-cpu.a llamacpp/build-cli/ggml/src/ggml-base.a
 *        -lm -lstdc++ -lpthread
 * Usage: sampler-probe <logits.txt> <seed0> <n_seeds> <temp> <top_k> <top_p> <min_p>
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: sampler-probe <logits> <seed0> <n_seeds> <temp> <top_k> <top_p> <min_p>\n");
        return 2;
    }
    std::vector<float> logits;
    FILE * f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    float v;
    while (fscanf(f, "%f", &v) == 1) {
        logits.push_back(v);
    }
    fclose(f);

    unsigned seed0 = (unsigned) atoi(argv[2]);
    int n_seeds = atoi(argv[3]);
    float temp = (float) atof(argv[4]);
    int32_t top_k = atoi(argv[5]);
    float top_p = (float) atof(argv[6]);
    float min_p = (float) atof(argv[7]);

    for (int s = 0; s < n_seeds; ++s) {
        std::vector<llama_token_data> data;
        data.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            data.push_back({(llama_token) i, logits[i], 0.0f});
        }
        llama_token_data_array arr = { data.data(), data.size(), -1, false };

        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed0 + s));
        llama_sampler_apply(chain, &arr);
        // selected is an index into the post-chain (filtered) array;
        // print the token id so it matches by1sample's ids[sel]
        printf("%u %lld\n", seed0 + s, (long long) arr.data[arr.selected].id);
        llama_sampler_free(chain);
    }
    return 0;
}
