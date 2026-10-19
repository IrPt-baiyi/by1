window.BY1_INDEX = [
 {
  "name": "DeepSeek-R1-Distill-Qwen-1.5B.by1",
  "slug": "DeepSeek-R1-Distill-Qwen-1.5B",
  "arch": "qwen2",
  "layers": 28,
  "d_model": 1536,
  "ctx": 131072,
  "n_tensors": 280,
  "mechs": [
   "FFN_",
   "GQA"
  ]
 },
 {
  "name": "DeepSeek-R1-Distill-Qwen-7B.by1",
  "slug": "DeepSeek-R1-Distill-Qwen-7B",
  "arch": "qwen2",
  "layers": 28,
  "d_model": 3584,
  "ctx": 131072,
  "n_tensors": 280,
  "mechs": [
   "FFN_",
   "GQA"
  ]
 },
 {
  "name": "GLM-4.7-Flash.by1",
  "slug": "GLM-4.7-Flash",
  "arch": "glm4_moe_lite",
  "layers": 48,
  "d_model": 2048,
  "ctx": 202752,
  "n_tensors": 707,
  "mechs": [
   "Dense",
   "MLA1",
   "MoE"
  ]
 },
 {
  "name": "GLM-5.3-Flash.by1",
  "slug": "GLM-5.3-Flash",
  "arch": "glm5_next",
  "layers": 70,
  "d_model": 4096,
  "ctx": 1048576,
  "n_tensors": 1311,
  "mechs": [
   "Dense",
   "KDA",
   "MoE",
   "SparseMLA",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Instella-3B.by1",
  "slug": "Instella-3B",
  "arch": "instella",
  "layers": 36,
  "d_model": 2560,
  "ctx": 4096,
  "n_tensors": 324,
  "mechs": [
   "Attn",
   "FFN_"
  ]
 },
 {
  "name": "Laguna-XS-2_1.by1",
  "slug": "Laguna-XS-2_1",
  "arch": "laguna",
  "layers": 40,
  "d_model": 2048,
  "ctx": 262144,
  "n_tensors": 595,
  "mechs": [
   "Dense",
   "GQA",
   "MoE"
  ]
 },
 {
  "name": "Ling-3.0-tiny.by1",
  "slug": "Ling-3.0-tiny",
  "arch": "bailing_hybrid",
  "layers": 24,
  "d_model": 1536,
  "ctx": 131072,
  "n_tensors": 469,
  "mechs": [
   "Dense",
   "KDA",
   "MLA",
   "MoE"
  ]
 },
 {
  "name": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.by1",
  "slug": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
  "arch": "nemotron_h",
  "layers": 54,
  "d_model": 2688,
  "ctx": 262144,
  "n_tensors": 356,
  "mechs": [
   "Attn",
   "MTPAttn",
   "MTPMoe",
   "Mamba",
   "MoE"
  ]
 },
 {
  "name": "Qwen2.5-0.5B-Instruct.by1",
  "slug": "Qwen2.5-0.5B-Instruct",
  "arch": "qwen2",
  "layers": 24,
  "d_model": "896",
  "ctx": 32768.0,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN"
  ]
 },
 {
  "name": "Qwen3-Coder-Next.by1",
  "slug": "Qwen3-Coder-Next",
  "arch": "qwen3_next",
  "layers": 48,
  "d_model": 2048,
  "ctx": 262144,
  "n_tensors": 708,
  "mechs": [
   "GQA",
   "Lin",
   "MoE"
  ]
 },
 {
  "name": "Qwen3-Omni-30B-A3B-Thinking.by1",
  "slug": "Qwen3-Omni-30B-A3B-Thinking",
  "arch": "qwen3_omni_moe",
  "layers": 107,
  "d_model": 2048,
  "ctx": 65536,
  "n_tensors": 1080,
  "mechs": [
   "AudioAttn",
   "AudioFfn",
   "GQA",
   "MoE",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Qwen3-VL-235B-A22B-Thinking.by1",
  "slug": "Qwen3-VL-235B-A22B-Thinking",
  "arch": "qwen3_vl_moe",
  "layers": 121,
  "d_model": 4096,
  "ctx": 262144,
  "n_tensors": 1062,
  "mechs": [
   "GQA",
   "MoE",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Qwen3-VL-2B-Thinking.by1",
  "slug": "Qwen3-VL-2B-Thinking",
  "arch": "qwen3_vl",
  "layers": 52,
  "d_model": 2048,
  "ctx": 262144,
  "n_tensors": 444,
  "mechs": [
   "FFN_",
   "GQA",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Qwen3-VL-30B-A3B-Thinking.by1",
  "slug": "Qwen3-VL-30B-A3B-Thinking",
  "arch": "qwen3_vl_moe",
  "layers": 75,
  "d_model": 2048,
  "ctx": 262144,
  "n_tensors": 648,
  "mechs": [
   "GQA",
   "MoE",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Qwen3.5-0.8B.by1",
  "slug": "Qwen3.5-0.8B",
  "arch": "qwen3_5",
  "layers": 37,
  "d_model": 1024,
  "ctx": 262144,
  "n_tensors": 372,
  "mechs": [
   "FFN_",
   "GQA",
   "Lin",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Qwen3.5-397B-A17B.by1",
  "slug": "Qwen3.5-397B-A17B",
  "arch": "qwen3_5_moe",
  "layers": 88,
  "d_model": 4096,
  "ctx": 262144,
  "n_tensors": 1145,
  "mechs": [
   "GQA",
   "Lin",
   "MTPAttn",
   "MTPMoe",
   "MoE",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Qwen3.5-9B.by1",
  "slug": "Qwen3.5-9B",
  "arch": "qwen3_5",
  "layers": 60,
  "d_model": 4096,
  "ctx": 262144,
  "n_tensors": 582,
  "mechs": [
   "FFN_",
   "GQA",
   "Lin",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Qwen3.6-35B-A3B.by1",
  "slug": "Qwen3.6-35B-A3B",
  "arch": "qwen3_5_moe",
  "layers": 68,
  "d_model": 2048,
  "ctx": 262144,
  "n_tensors": 839,
  "mechs": [
   "GQA",
   "Lin",
   "MTPAttn",
   "MTPMoe",
   "MoE",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Qwen3.8-27B-FP8.by1",
  "slug": "Qwen3.8-27B-FP8",
  "arch": "qwen3_5",
  "layers": 65,
  "d_model": "5120",
  "ctx": 262144.0,
  "n_tensors": 404,
  "mechs": [
   "A",
   "D",
   "F"
  ]
 },
 {
  "name": "Qwen3.8-27B.by1",
  "slug": "Qwen3.8-27B",
  "arch": "qwen3_5",
  "layers": 92,
  "d_model": 5120,
  "ctx": 262144,
  "n_tensors": 945,
  "mechs": [
   "FFN_",
   "GQA",
   "Lin",
   "MTPAttn",
   "MTPFfn",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Qwen3.8-Flash-Next.by1",
  "slug": "Qwen3.8-Flash-Next",
  "arch": "qwen4_exp",
  "layers": 76,
  "d_model": 2560,
  "ctx": 262144,
  "n_tensors": 993,
  "mechs": [
   "GQA",
   "Lin",
   "MoE",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "Step-3.7-Flash.by1",
  "slug": "Step-3.7-Flash",
  "arch": "step3p5",
  "layers": 95,
  "d_model": 4096,
  "ctx": 262144,
  "n_tensors": 931,
  "mechs": [
   "Attn",
   "Dense",
   "MoE",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "attn-fused-shaped.by1",
  "slug": "attn-fused-shaped",
  "arch": "",
  "layers": 2,
  "d_model": 128,
  "ctx": 512,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN_"
  ]
 },
 {
  "name": "attn-qgate-shaped.by1",
  "slug": "attn-qgate-shaped",
  "arch": "",
  "layers": 2,
  "d_model": 128,
  "ctx": 512,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN_"
  ]
 },
 {
  "name": "clef-tiny.by1",
  "slug": "clef-tiny",
  "arch": "qwen3_5",
  "layers": 8,
  "d_model": 256,
  "ctx": 262144,
  "n_tensors": 90,
  "mechs": [
   "FFN_",
   "GQA",
   "Lin"
  ]
 },
 {
  "name": "clef.by1",
  "slug": "clef",
  "arch": "qwen3_5",
  "layers": 91,
  "d_model": 5120,
  "ctx": 262144,
  "n_tensors": 936,
  "mechs": [
   "FFN_",
   "GQA",
   "Lin",
   "Vis",
   "VisMlp"
  ]
 },
 {
  "name": "complete-Falcon-H1-0.5B-Instruct.by1",
  "slug": "complete-Falcon-H1-0.5B-Instruct",
  "arch": "falcon_h1",
  "layers": 36,
  "d_model": "1024",
  "ctx": 16384.0,
  "n_tensors": 0,
  "mechs": [
   "Attn",
   "Both",
   "MLP",
   "Mixer"
  ]
 },
 {
  "name": "complete-Qwen2.5-7B-Instruct.by1",
  "slug": "complete-Qwen2.5-7B-Instruct",
  "arch": "qwen2",
  "layers": 28,
  "d_model": "3584",
  "ctx": 32768.0,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN"
  ]
 },
 {
  "name": "complete-Qwen3-0.6B.by1",
  "slug": "complete-Qwen3-0.6B",
  "arch": "qwen3",
  "layers": 28,
  "d_model": 1024,
  "ctx": 40960,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN"
  ]
 },
 {
  "name": "complete-Qwen3-8B.by1",
  "slug": "complete-Qwen3-8B",
  "arch": "qwen3",
  "layers": 36,
  "d_model": 4096,
  "ctx": 40960,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN"
  ]
 },
 {
  "name": "complete-Qwen3.8-27B.by1",
  "slug": "complete-Qwen3.8-27B",
  "arch": "qwen3_5",
  "layers": 92,
  "d_model": "5120",
  "ctx": 262144.0,
  "n_tensors": 0,
  "mechs": [
   "A",
   "D",
   "F",
   "V",
   "VF"
  ]
 },
 {
  "name": "complete-gpt-oss-20b.by1",
  "slug": "complete-gpt-oss-20b",
  "arch": "gpt_oss",
  "layers": 24,
  "d_model": 2880,
  "ctx": 131072,
  "n_tensors": 0,
  "mechs": [
   "A",
   "MoE"
  ]
 },
 {
  "name": "complete-gpt2.by1",
  "slug": "complete-gpt2",
  "arch": "gpt2",
  "layers": 12,
  "d_model": 768,
  "ctx": 1024,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN"
  ]
 },
 {
  "name": "complete-granite-4.0-h-1b.by1",
  "slug": "complete-granite-4.0-h-1b",
  "arch": "granitemoehybrid",
  "layers": 40,
  "d_model": "1536",
  "ctx": 131072.0,
  "n_tensors": 0,
  "mechs": [
   "Attn",
   "MLP",
   "Mixer"
  ]
 },
 {
  "name": "complete-granite-4.0-h-350m.by1",
  "slug": "complete-granite-4.0-h-350m",
  "arch": "granitemoehybrid",
  "layers": 32,
  "d_model": "768",
  "ctx": 32768.0,
  "n_tensors": 0,
  "mechs": [
   "Attn",
   "MLP",
   "Mixer"
  ]
 },
 {
  "name": "complete-mamba-130m-hf.by1",
  "slug": "complete-mamba-130m-hf",
  "arch": "mamba",
  "layers": 24,
  "d_model": "768",
  "ctx": 2048.0,
  "n_tensors": 0,
  "mechs": [
   "Mixer"
  ]
 },
 {
  "name": "distill-student.by1",
  "slug": "distill-student",
  "arch": "",
  "layers": 2,
  "d_model": 64,
  "ctx": 128,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN"
  ]
 },
 {
  "name": "distill-teacher.by1",
  "slug": "distill-teacher",
  "arch": "",
  "layers": 2,
  "d_model": 128,
  "ctx": 128,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN"
  ]
 },
 {
  "name": "embeddinggemma-2.by1",
  "slug": "embeddinggemma-2",
  "arch": "embedding_gemma2",
  "layers": 52,
  "d_model": 512,
  "ctx": 262144,
  "n_tensors": 912,
  "mechs": [
   "AudioAttn",
   "AudioFFN",
   "Dense",
   "GQA",
   "ViT",
   "ViTMlp"
  ]
 },
 {
  "name": "gate-probe.by1",
  "slug": "gate-probe",
  "arch": "probe",
  "layers": 2,
  "d_model": "64",
  "ctx": 512.0,
  "n_tensors": 14,
  "mechs": [
   "Attn",
   "FFN_"
  ]
 },
 {
  "name": "gelu-shaped.by1",
  "slug": "gelu-shaped",
  "arch": "",
  "layers": 1,
  "d_model": 8,
  "ctx": 128,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "MLP"
  ]
 },
 {
  "name": "gemma-4-12B.by1",
  "slug": "gemma-4-12B",
  "arch": "gemma4_unified",
  "layers": 48,
  "d_model": 3840,
  "ctx": 262144,
  "n_tensors": 432,
  "mechs": [
   "Dense",
   "GQA"
  ]
 },
 {
  "name": "gemma-4-26B-A4B.by1",
  "slug": "gemma-4-26B-A4B",
  "arch": "gemma4",
  "layers": 57,
  "d_model": 2816,
  "ctx": 262144,
  "n_tensors": 573,
  "mechs": [
   "GQA",
   "MoE",
   "ViT",
   "ViTMlp"
  ]
 },
 {
  "name": "gemma-4-31B.by1",
  "slug": "gemma-4-31B",
  "arch": "gemma4",
  "layers": 87,
  "d_model": 5376,
  "ctx": 262144,
  "n_tensors": 783,
  "mechs": [
   "Dense",
   "GQA",
   "ViT",
   "ViTMlp"
  ]
 },
 {
  "name": "gemma-4-E4B.by1",
  "slug": "gemma-4-E4B",
  "arch": "gemma4",
  "layers": 70,
  "d_model": 2560,
  "ctx": 131072,
  "n_tensors": 1522,
  "mechs": [
   "AudioAttn",
   "AudioFFN",
   "Dense",
   "GQA",
   "ViT",
   "ViTMlp"
  ]
 },
 {
  "name": "glm-text-tiny.by1",
  "slug": "glm-text-tiny",
  "arch": "glm5_next",
  "layers": 4,
  "d_model": 256,
  "ctx": 1048576,
  "n_tensors": 0,
  "mechs": [
   "Dense",
   "KDA",
   "MoE",
   "SparseMLA"
  ]
 },
 {
  "name": "gpt-oss-120b.by1",
  "slug": "gpt-oss-120b",
  "arch": "gpt_oss",
  "layers": 36,
  "d_model": 2880,
  "ctx": 131072,
  "n_tensors": 612,
  "mechs": [
   "GQA",
   "MoE"
  ]
 },
 {
  "name": "gpt-oss-20b.by1",
  "slug": "gpt-oss-20b",
  "arch": "gpt_oss",
  "layers": 24,
  "d_model": 2880,
  "ctx": 131072,
  "n_tensors": 408,
  "mechs": [
   "GQA",
   "MoE"
  ]
 },
 {
  "name": "gpt-oss-shaped.by1",
  "slug": "gpt-oss-shaped",
  "arch": "",
  "layers": 2,
  "d_model": 64,
  "ctx": 512,
  "n_tensors": 0,
  "mechs": [
   "Experts",
   "GQA"
  ]
 },
 {
  "name": "gpt2-tiny.by1",
  "slug": "gpt2-tiny",
  "arch": "gpt2",
  "layers": 2,
  "d_model": 64,
  "ctx": 64,
  "n_tensors": 18,
  "mechs": [
   "Attn",
   "MLP"
  ]
 },
 {
  "name": "gpt2.by1",
  "slug": "gpt2",
  "arch": "gpt2",
  "layers": 12,
  "d_model": 768,
  "ctx": 1024,
  "n_tensors": 108,
  "mechs": [
   "Attn",
   "MLP"
  ]
 },
 {
  "name": "hello.by1",
  "slug": "hello",
  "arch": "",
  "layers": 4,
  "d_model": 128,
  "ctx": 256,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "FFN"
  ]
 },
 {
  "name": "kda-gva-shaped.by1",
  "slug": "kda-gva-shaped",
  "arch": "kda_gva_shaped",
  "layers": 4,
  "d_model": 128,
  "ctx": 1024,
  "n_tensors": 64,
  "mechs": [
   "KDA_",
   "MLP_"
  ]
 },
 {
  "name": "kda-lowrank-shaped.by1",
  "slug": "kda-lowrank-shaped",
  "arch": "kda_lowrank_shaped",
  "layers": 4,
  "d_model": 128,
  "ctx": 1024,
  "n_tensors": 76,
  "mechs": [
   "KDA_",
   "MLP_"
  ]
 },
 {
  "name": "kda-shaped.by1",
  "slug": "kda-shaped",
  "arch": "kda_shaped",
  "layers": 4,
  "d_model": 128,
  "ctx": 1024,
  "n_tensors": 64,
  "mechs": [
   "KDA_",
   "MLP_"
  ]
 },
 {
  "name": "laguna-tiny.by1",
  "slug": "laguna-tiny",
  "arch": "laguna",
  "layers": 2,
  "d_model": 64,
  "ctx": 4096,
  "n_tensors": 0,
  "mechs": [
   "Dense",
   "GQA"
  ]
 },
 {
  "name": "linear-gdn-shaped.by1",
  "slug": "linear-gdn-shaped",
  "arch": "",
  "layers": 2,
  "d_model": 64,
  "ctx": 512,
  "n_tensors": 0,
  "mechs": [
   "FFN_",
   "GDN"
  ]
 },
 {
  "name": "ling-tiny.by1",
  "slug": "ling-tiny",
  "arch": "bailing_hybrid",
  "layers": 4,
  "d_model": 256,
  "ctx": 262144,
  "n_tensors": 74,
  "mechs": [
   "Dense",
   "KDA",
   "MLA",
   "MoE"
  ]
 },
 {
  "name": "llama-shaped.by1",
  "slug": "llama-shaped",
  "arch": "",
  "layers": 4,
  "d_model": 128,
  "ctx": 512,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "MLP"
  ]
 },
 {
  "name": "llama3-shaped.by1",
  "slug": "llama3-shaped",
  "arch": "llama3_shaped",
  "layers": 4,
  "d_model": 256,
  "ctx": 262144,
  "n_tensors": 28,
  "mechs": [
   "Full",
   "MLP_",
   "Slide"
  ]
 },
 {
  "name": "mamba-130m-hf.by1",
  "slug": "mamba-130m-hf",
  "arch": "mamba",
  "layers": 24,
  "d_model": "768",
  "ctx": 2048.0,
  "n_tensors": 216,
  "mechs": [
   "Mixer"
  ]
 },
 {
  "name": "minimind-3.by1",
  "slug": "minimind-3",
  "arch": "qwen3",
  "layers": 8,
  "d_model": 768,
  "ctx": 32768,
  "n_tensors": 72,
  "mechs": [
   "Attn",
   "FFN_"
  ]
 },
 {
  "name": "mixtral-shaped.by1",
  "slug": "mixtral-shaped",
  "arch": "",
  "layers": 4,
  "d_model": 128,
  "ctx": 512,
  "n_tensors": 0,
  "mechs": [
   "Attention",
   "Experts"
  ]
 },
 {
  "name": "mla-shaped.by1",
  "slug": "mla-shaped",
  "arch": "mla_shaped",
  "layers": 4,
  "d_model": 256,
  "ctx": 1024,
  "n_tensors": 44,
  "mechs": [
   "MLA_",
   "MLP_"
  ]
 },
 {
  "name": "mla-sparse-shaped.by1",
  "slug": "mla-sparse-shaped",
  "arch": "mla_sparse_shaped",
  "layers": 4,
  "d_model": 256,
  "ctx": 1024,
  "n_tensors": 68,
  "mechs": [
   "MLAK_",
   "MLA_",
   "MLP_"
  ]
 },
 {
  "name": "moe-routing-shaped.by1",
  "slug": "moe-routing-shaped",
  "arch": "",
  "layers": 3,
  "d_model": 128,
  "ctx": 512,
  "n_tensors": 0,
  "mechs": [
   "Attn",
   "MoEA",
   "MoEB",
   "MoEC"
  ]
 },
 {
  "name": "multibranch-tiny.by1",
  "slug": "multibranch-tiny",
  "arch": "qwen3_5",
  "layers": 7,
  "d_model": "64",
  "ctx": 4096.0,
  "n_tensors": 0,
  "mechs": [
   "A",
   "D",
   "F",
   "V",
   "VF"
  ]
 },
 {
  "name": "qwen3-next-shaped.by1",
  "slug": "qwen3-next-shaped",
  "arch": "",
  "layers": 4,
  "d_model": 64,
  "ctx": 512,
  "n_tensors": 0,
  "mechs": [
   "GDN",
   "GQA",
   "MoE"
  ]
 },
 {
  "name": "raw-escape.by1",
  "slug": "raw-escape",
  "arch": "probe",
  "layers": 2,
  "d_model": 64,
  "ctx": 256,
  "n_tensors": 8,
  "mechs": [
   "FFN_",
   "Weird"
  ]
 },
 {
  "name": "selftest.by1",
  "slug": "selftest",
  "arch": "",
  "layers": 9,
  "d_model": "1024",
  "ctx": 8192.0,
  "n_tensors": 4,
  "mechs": [
   "FFN",
   "Global",
   "Local"
  ]
 },
 {
  "name": "ssm-shaped.by1",
  "slug": "ssm-shaped",
  "arch": "ssm_shaped",
  "layers": 4,
  "d_model": 128,
  "ctx": 1024,
  "n_tensors": 44,
  "mechs": [
   "MLP_",
   "Mamba_"
  ]
 }
];
