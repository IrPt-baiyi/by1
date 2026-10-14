/* mtmd-embd-probe.cpp -- T20 judge: vision tower output via mtmd.
 *
 * Loads an mmproj GGUF + a raw RGB image (width/height from args), runs
 * tokenize -> mtmd_encode_chunk, dumps the image-token embeddings as text
 * ("embd n_tokens n_per_token" header + one float per line). The
 * transformers side (tools/tiny-vision-ref.py) produces the same boundary
 * for comparison.
 *
 * Build (from repo root):
 *   g++ -O2 -fopenmp -o tools/mtmd-embd-probe.exe tools/mtmd-embd-probe.cpp \
 *       -Illamacpp/llama.cpp/include -Illamacpp/llama.cpp/tools/mtmd \
 *       -Illamacpp/llama.cpp/ggml/include \
 *       llamacpp/build-cli/tools/mtmd/libmtmd.a \
 *       llamacpp/build-cli/src/libllama.a \
 *       llamacpp/build-cli/ggml/src/ggml.a \
 *       llamacpp/build-cli/ggml/src/ggml-cpu.a \
 *       llamacpp/build-cli/ggml/src/ggml-base.a \
 *       -lm -lstdc++ -lpthread
 *
 * Usage: mtmd-embd-probe <mmproj.gguf> <image.rgb> <nx> <ny> <marker> <n_per_token> <text.gguf>
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: mtmd-embd-probe <mmproj> <image.rgb> <nx> <ny> <marker>\n");
        return 2;
    }
    const char * mmproj = argv[1];
    const char * img_path = argv[2];
    uint32_t nx = (uint32_t) atoi(argv[3]);
    uint32_t ny = (uint32_t) atoi(argv[4]);
    const char * marker = argv[5];

    FILE * f = fopen(img_path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", img_path);
        return 2;
    }
    std::vector<unsigned char> rgb((size_t) nx * ny * 3);
    if (fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) {
        fprintf(stderr, "short read on %s\n", img_path);
        return 2;
    }
    fclose(f);

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false;
    params.media_marker = argv[5];   // marker is caller-supplied (same string in text)
    // tokenize needs the text model's vocab (mtmd_init_from_file takes the model)
    struct llama_model * text_model =
        llama_model_load_from_file(argv[7], llama_model_default_params());
    if (!text_model) {
        fprintf(stderr, "llama_model_load_from_file failed\n");
        return 1;
    }
    mtmd_context * ctx = mtmd_init_from_file(mmproj, text_model, params);
    if (!ctx) {
        fprintf(stderr, "mtmd_init_from_file failed\n");
        return 1;
    }

    mtmd_bitmap * bmp = mtmd_bitmap_init(nx, ny, rgb.data());
    std::vector<const mtmd_bitmap *> bmps = { bmp };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text text = { marker, strlen(marker), false, true };
    int rc = mtmd_tokenize(ctx, chunks, &text, bmps.data(), bmps.size());
    if (rc != 0) {
        fprintf(stderr, "mtmd_tokenize failed rc=%d (marker=%s)\n", rc, marker);
        return 1;
    }

    int32_t enc_rc = -1;
    size_t n_tok = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            continue;
        }
        enc_rc = mtmd_encode_chunk(ctx, (mtmd_input_chunk *) ch);
        n_tok = mtmd_input_chunk_get_n_tokens(ch);
        break;
    }
    if (enc_rc != 0) {
        fprintf(stderr, "mtmd_encode_chunk failed rc=%d\n", enc_rc);
        return 1;
    }
    const float * embd = mtmd_get_output_embd(ctx);
    size_t n_per = (size_t) atoi(argv[6]);
    printf("embd %zu %zu\n", n_tok, n_per);
    for (size_t i = 0; i < n_tok * n_per; ++i) {
        printf("%.9g\n", (double) embd[i]);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(ctx);
    return 0;
}
