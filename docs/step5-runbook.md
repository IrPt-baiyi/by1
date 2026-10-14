# Step-5 快速反应 runbook（发布日即跑）

> 目的：Step-5 发布后**第一时间**用 by1 工具链出验证结论（也是推销素材）。
> 命令全部可直接复制；判据与既有 Step 系（stepfun-ai/Step-3.7-Flash refs 在库）同款。

## 0. 前置（现在已就绪）

- v1.0.0-rc1 已推（档 3 全绿 100 项 / 档 5 全绿 129 项）。
- Step 家族机制（sigmoid_topk 路由、滑窗注意力、逐层数组周期续）已在
  by1codegen/by1exec/by1export 里有实现与判例。
- refs 语料工具链：`by1refs` / `by1sync` / `by1llama manifest-audit`。

## 1. 发布日 · 抓元数据（~5 分钟）

```powershell
# 找官方 repo（先确认是 stepfun-ai/Step-5 还是别的命名）
python src/by1refs.py --fetch <owner>/<Repo>     # config + tensors 清单入 refs/
```

判据：`refs/<owner>__<Repo>.config.json` 与 `.tensors.json` 落库；记录 **revision**（sha）。

## 2. 草稿 .by1（~10 分钟）

```powershell
python src/by1boot.py refs/<owner>__<Repo>.config.json refs/<owner>__<Repo>.tensors.json --name Step-5 --out models/<Step-5>.by1
python src/by1check.py models/<Step-5>.by1        # 必须 0 错误；有错逐条修
```

判据：by1check 0/0/0；`by1refs` 双向对拍无缺项。

## 3. 合同审计（~5 分钟）

```powershell
python src/by1check.py models/<Step-5>.by1 --emit-config build/step5-cfg.json
python src/checks/by1sync.py                    # config 逐字段 + 张量名/形状对账
```

判据：字段逐项有账；缺失/多余各有清单（不许静默补默认）。

## 4. GGUF 产物线（~10 分钟，需权重下载后）

```powershell
# 权重下载（HF）后：
python src/checks/by1llama.py build-gguf <hf-dir> build/step5.gguf --ref <上游 convert 产物>
.\llamacpp\build-cli\bin\llama-eval-logits.exe -m build\step5.gguf -p "The capital of France is" > build\step5.logits
```

判据：张量区逐字节 vs 上游 convert 产物；logits 逐位（T23 判据同款）。

## 5. 推销素材（当天出）

- 一张表：`.by1` 字段 ↔ config ↔ tensor manifest ↔ GGUF 四方一致（数字）。
- 失败/跳过单列（诚实是卖点）。
- 与 Step-3.7 官方 refs 的机制对照（sigmoid_topk 等）——证明"同族新模型零新机制"。

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| 新机制（by1boot 不认） | 按"要么实现要么拒绝"记录缺口；先出合同差异表，不硬圆 |
| llama.cpp 未支持新结构 | 记录上游 issue/PR 状态；产物线仍可出（加载判据降级为 gguf-check） |
| 权重太大（>100B） | 先用 config/tensor 合同 + 单层权重抽样审计，推理判据标注范围 |
