window.BY1_INDEX = [
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
 }
];
