/* audio-mel-probe.cpp -- T21 judge: audio preprocessing boundary (mtmd side).
 *
 * Loads an audio mmproj, builds the mtmd_audio_preprocessor from its
 * hparams, feeds a synthetic sine (440 Hz, the same recipe as
 * llama-mtmd-debug --audio 440), and dumps the mel spectrogram as text
 * ("mel n_mel n_len" header + one float per line, mel-major [m][t]).
 * The transformers side (tools/qwen3asr-mel-ref.py) produces the same
 * boundary for comparison.
 *
 * Build (from repo root):
 *   g++ -O2 -fopenmp -o tools/audio-mel-probe.exe tools/audio-mel-probe.cpp \
 *       -Illamacpp/llama.cpp/include -Illamacpp/llama.cpp/tools/mtmd \
 *       -Illamacpp/llama.cpp/ggml/include \
 *       llamacpp/build-cli/tools/mtmd/libmtmd.a \
 *       llamacpp/build-cli/src/libllama.a \
 *       llamacpp/build-cli/ggml/src/ggml.a \
 *       llamacpp/build-cli/ggml/src/ggml-cpu.a \
 *       llamacpp/build-cli/ggml/src/ggml-base.a \
 *       -lm -lstdc++ -lpthread
 *
 * Usage: audio-mel-probe <mmproj.gguf> <n_samples>
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "clip.h"
#include "mtmd-audio.h"

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: audio-mel-probe <mmproj> <n_samples>\n");
        return 2;
    }
    clip_context_params cparams = {};
    cparams.image_min_tokens = -1;
    cparams.image_max_tokens = -1;
    fprintf(stderr, "step: clip_init...\n");
    clip_init_result res = clip_init(argv[1], cparams);
    fprintf(stderr, "step: clip_init done ctx_a=%p\n", (void *) res.ctx_a);
    if (!res.ctx_a) {
        fprintf(stderr, "no audio context in %s\n", argv[1]);
        return 1;
    }
    const clip_hparams * hp = clip_get_hparams(res.ctx_a);
    fprintf(stderr, "step: hparams sr=%d\n", hp->audio_sample_rate);
    int n = atoi(argv[2]);
    /* the llama-mtmd-debug "440" recipe: sine at 440 Hz over the sample rate */
    const int sample_rate = hp->audio_sample_rate > 0 ? hp->audio_sample_rate
                                                     : 16000;
    std::vector<float> pcm(n);
    for (int i = 0; i < n; ++i) {
        pcm[i] = std::sin(2.0 * 3.14159265f * 440.0f * i / sample_rate);
    }
    fprintf(stderr, "step: preprocessor ctor (qwen3a)...\n");
    mtmd_audio_preprocessor_qwen3a pre(res.ctx_a);
    pre.initialize();
    fprintf(stderr, "step: preprocess %d samples...\n", n);
    std::vector<mtmd_audio_mel> chunks;
    if (!pre.preprocess(pcm.data(), (size_t) n, chunks)) {
        fprintf(stderr, "preprocess failed\n");
        return 1;
    }
    fprintf(stderr, "step: preprocess done chunks=%zu\n", chunks.size());
    size_t n_mel = 0, n_len = 0, total = 0;
    for (auto & c : chunks) {
        n_mel = (size_t) c.n_mel;
        n_len += (size_t) c.n_len;
        total += (size_t) c.n_mel * (size_t) c.n_len;
    }
    printf("mel %zu %zu\n", n_mel, n_len);
    for (auto & c : chunks) {
        for (int m = 0; m < c.n_mel; ++m) {
            for (int t = 0; t < c.n_len; ++t) {
                printf("%.9g\n", (double) c.data[(size_t) m * c.n_len + t]);
            }
        }
    }
    clip_free(res.ctx_v);
    clip_free(res.ctx_a);
    return 0;
}
