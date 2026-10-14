/* llama-eval-logits.cpp -- fixed prompt -> full logits dump (one float per line).
 *
 * T18 chain's dump producer. by1llama._load_logits accepts "one number per
 * line"; llama-eval-callback prints truncated debug previews, so bitwise
 * compare is impossible on its output. This tool prints the full final-step
 * logits vector at full float precision.
 *
 * Build (from repo root):
 *   gcc -O2 -fopenmp -o llamacpp/llama-eval-logits.exe llamacpp/llama-eval-logits.cpp \
 *       -Illamacpp/llama.cpp/include -Illamacpp/llama.cpp/common \
 *       -Illamacpp/llama.cpp/ggml/include \
 *       llamacpp/build-cli/common/libllama-common.a \
 *       llamacpp/build-cli/src/libllama.a \
 *       llamacpp/build-cli/ggml/src/ggml-cpu.a \
 *       llamacpp/build-cli/ggml/src/ggml.a \
 *       llamacpp/build-cli/ggml/src/ggml-base.a \
 *       -lm -lstdc++ -lpthread
 *
 * Usage:
 *   llama-eval-logits -m model.gguf -p "prompt" > dump.logits
 */
#include <clocale>
#include <cstdio>
#include <string>
#include <vector>

#include "arg.h"
#include "common.h"
#include "llama.h"

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    fprintf(stderr, "tokens(%zu):", tokens.size());
    for (auto t : tokens) {
        fprintf(stderr, " %d", t);
    }
    fprintf(stderr, "\n");
    llama_batch batch = llama_batch_get_one(tokens.data(), (int) tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, (int) tokens.size() - 1);
    for (int i = 0; i < n_vocab; ++i) {
        printf("%.9g\n", (double) logits[i]);
    }

    llama_backend_free();
    return 0;
}
