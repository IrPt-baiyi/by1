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
#include <cmath>
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
    /* audio mode: <mmproj> --sine <n_samples> <marker> <n_per> <text.gguf> */
    bool audio_mode = (argc >= 7 && std::string(argv[2]) == "--sine");
    const char * marker = audio_mode ? argv[4] : argv[5];
    const char * n_per_arg = audio_mode ? argv[5] : argv[6];
    const char * text_model_path = audio_mode ? argv[6] : argv[7];
    std::vector<unsigned char> rgb;
    std::vector<float> pcm;
    uint32_t nx = 0, ny = 0;
    if (audio_mode) {
        int ns = atoi(argv[3]);
        pcm.resize(ns);
        for (int i = 0; i < ns; ++i) {
            pcm[i] = std::sin(2.0 * 3.14159265 * 440.0 * i / 16000.0);
        }
    } else {
        nx = (uint32_t) atoi(argv[3]);
        ny = (uint32_t) atoi(argv[4]);
        FILE * f = fopen(argv[2], "rb");
        if (!f) {
            fprintf(stderr, "cannot open %s\n", argv[2]);
            return 2;
        }
        rgb.resize((size_t) nx * ny * 3);
        if (fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) {
            fprintf(stderr, "short read on %s\n", argv[2]);
            return 2;
        }
        fclose(f);
    }

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false;
    params.media_marker = marker;   // marker is caller-supplied (same string in text)
    // tokenize needs the text model's vocab (mtmd_init_from_file takes the model)
    struct llama_model * text_model =
        llama_model_load_from_file(text_model_path, llama_model_default_params());
    if (!text_model) {
        fprintf(stderr, "llama_model_load_from_file failed\n");
        return 1;
    }
    mtmd_context * ctx = mtmd_init_from_file(mmproj, text_model, params);
    if (!ctx) {
        fprintf(stderr, "mtmd_init_from_file failed\n");
        return 1;
    }

    mtmd_bitmap * bmp = audio_mode
        ? mtmd_bitmap_init_from_audio(pcm.size(), pcm.data())
        : mtmd_bitmap_init(nx, ny, rgb.data());
    std::vector<const mtmd_bitmap *> bmps = { bmp };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text text = { marker, strlen(marker), false, true };
    int rc = mtmd_tokenize(ctx, chunks, &text, bmps.data(), bmps.size());
    if (rc != 0) {
        fprintf(stderr, "mtmd_tokenize failed rc=%d (marker=%s)\n", rc, marker);
        return 1;
    }

    /* T21-11: dump the exact input token sequence (text chunks as ids,
       media chunk as M*n) so the HF side can replicate it 1:1. */
    fprintf(stderr, "tokens:");
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * ch2 = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch2) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t nt = 0;
            const llama_token * toks = mtmd_input_chunk_get_tokens_text(ch2, &nt);
            for (size_t j = 0; j < nt; ++j) fprintf(stderr, " %d", toks[j]);
        } else {
            size_t nt = mtmd_input_chunk_get_n_tokens(ch2);
            fprintf(stderr, " M*%zu", nt);
        }
    }
    fprintf(stderr, "\n");

    int32_t enc_rc = -1;
    size_t n_tok = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        mtmd_input_chunk_type cty = mtmd_input_chunk_get_type(ch);
        if (cty != MTMD_INPUT_CHUNK_TYPE_IMAGE &&
            cty != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
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
    size_t n_per = (size_t) atoi(n_per_arg);
    printf("embd %zu %zu\n", n_tok, n_per);
    for (size_t i = 0; i < n_tok * n_per; ++i) {
        printf("%.9g\n", (double) embd[i]);
    }
    /* T21 e2e judge: LM next-token logits over the exact input sequence.
       Media rows enter as embd entries (token=-1), text chunks as ids. */
    {
        int n_all = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk * ch3 = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(ch3) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t nt = 0;
                mtmd_input_chunk_get_tokens_text(ch3, &nt);
                n_all += (int) nt;
            } else {
                n_all += (int) mtmd_input_chunk_get_n_tokens(ch3);
            }
        }
        struct llama_context_params cparams = llama_context_default_params();
        struct llama_context * lctx = llama_new_context_with_model(text_model, cparams);
        if (!lctx) { fprintf(stderr, "lm ctx failed\n"); return 1; }
        /* T21 final: per-chunk API-correct calls -- text chunks go through
           llama_decode, the media (embd) chunk through llama_encode, which
           is exactly what mtmd_helper does (LLM-P / LLM-D split). */
        std::vector<llama_token>   s_tok;
        std::vector<float>         s_embd;
        std::vector<llama_pos>     s_pos;
        std::vector<int32_t>       s_nseq;
        std::vector<llama_seq_id>  s_seq;
        std::vector<llama_seq_id*> s_seqptr;
        std::vector<int8_t>        s_log;
        const float * embd_all = mtmd_get_output_embd(ctx);
        int slot = 0, med_row = 0, last_local = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk * ch3 = mtmd_input_chunks_get(chunks, i);
            int ty3 = (int) mtmd_input_chunk_get_type(ch3);
            size_t nt = 0;
            const llama_token * tk = NULL;
            if (ty3 == (int) MTMD_INPUT_CHUNK_TYPE_TEXT) {
                tk = mtmd_input_chunk_get_tokens_text(ch3, &nt);
            } else {
                nt = mtmd_input_chunk_get_n_tokens(ch3);
            }
            int cnt = (int) nt;
            s_tok.assign(cnt, 0);
            s_embd.assign((size_t) cnt * n_per, 0.0f);
            s_pos.assign(cnt, 0); s_nseq.assign(cnt, 1);
            s_seq.assign(cnt, 0); s_seqptr.assign(cnt, nullptr);
            s_log.assign(cnt, 0);
            for (int j = 0; j < cnt; ++j) {
                s_seqptr[j] = &s_seq[j];
                s_pos[j] = slot + j;
            }
            struct llama_batch lb;
            lb.n_tokens = cnt;
            lb.token = s_tok.data();
            lb.embd = s_embd.data();
            lb.pos = s_pos.data();
            lb.n_seq_id = s_nseq.data();
            lb.seq_id = s_seqptr.data();
            lb.logits = s_log.data();
            int is_last = (i == mtmd_input_chunks_size(chunks) - 1);
            if (is_last) { s_log[cnt - 1] = 1; last_local = cnt - 1; }
            if (ty3 == (int) MTMD_INPUT_CHUNK_TYPE_TEXT) {
                for (int j = 0; j < cnt; ++j) lb.token[j] = tk[j];
                fprintf(stderr, "lm: decode text chunk %zu n=%d\n", i, cnt);
                if (llama_decode(lctx, lb) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            } else {
                for (int j = 0; j < cnt; ++j) {
                    memcpy(lb.embd + (size_t) j * n_per,
                           embd_all + (size_t) med_row * n_per,
                           sizeof(float) * n_per);
                    med_row++;
                }
                /* audio media block is CAUSAL (non-causal was refuted:
                   cos 0.579 vs causal 0.705; scope_non_causal belongs to
                   image chunks). Keep plain causal decode. */
                fprintf(stderr, "lm: decode media chunk %zu n=%d (causal)\n", i, cnt);
                if (llama_decode(lctx, lb) != 0) { fprintf(stderr, "media decode failed\n"); return 1; }
            }
            slot += cnt;
        }
        const float * lg = llama_get_logits_ith(lctx, last_local);
        int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(text_model));
        printf("lm %d\n", n_vocab);
        for (int i = 0; i < n_vocab; ++i) printf("%.9g\n", (double) lg[i]);
        llama_free(lctx);
    }

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(ctx);
    return 0;
}
