# 架构浏览器计划（2026-10-18）

> 一句话：把「`.by1` 描述 → IR → 层结构 → 原语图」摊开成**可浏览的事实**。
> 它不新增判据，而是让已有判据与结构**看得见** —— 上一轮就是靠它一眼看出
> `laguna-tiny` 有 0 层（B3 假绿）。

## 〇、原则

1. **唯一真相**：页面数据一律来自 `by1check` / `by1codegen` / `by1graph`，
   页面不自己解析 `.by1`，也不维护第二份数字。
2. **只读**：站点不写回 IR、不做编辑器；显示错了要修视图或修 IR，不修文档。
3. **可复核**：`python tools/by1site.py --check` 把磁盘数据与现算 IR 逐字节比；
   漂移必须红。
4. **纯静态**：双击 `site/index.html` 就能用，不起服务器、不引构建链、
   数据用 `window.BY1_*` 脚本而不是 `fetch(json)`（file:// 的 CORS 限制）。

## 一、里程碑

| # | 内容 | 判据 | 状态 |
|---|---|---|---|
| M1 | 导出器 + 结构前端（10 份代表模型）+ `--check` | `--check` 绿；改一个数字 -> 红 | ✅ 已完成（提交 `1cbb373`） |
| M2 | 全量导出（`--all`，约 50 份）+ `--check` 进 `by1all` 门禁 | 门禁里有一行 `[PASS] 站点数据与 IR 一致：N 份`；手改数据 -> 档 3/5 红 | ✅ 已完成：68 份建索引（53 份导出成功，15 份部分描述/多分支如实标红）；`by1sitedata.py --check` 已进档 3（tier 3 = 119 项），齿齿：改一个数字 -> rc=1 并报 "gelu-shaped 与 IR 不一致" |
| M3 | **图视图**：用 `graphs-golden.json` 的 nodes/edges 画原语图 | 点一层能看到它 lower 出的节点；节点数与 golden 对得上 | 待做 |
| M4 | **模型对比**：选两份模型 diff IR（层数 / 机制 / 属性 / 张量） | 对比页给出的差异条数与现算 diff 一致 | 待做 |
| M5 | （可选）证据叠加：门禁判定行、config 对账数字 | 页面数字与 `by1all` 输出逐条一致 | 视需要 |

## 二、M2 的具体做法

1. `python tools/by1site.py --all --out site/data` —— 导不出的模型（缺 `vocab` 的部分描述）
   **如实列进索引并标红**，不静默丢；
2. 在 `src/checks/by1all.py` 的判卷人表里加一行 `by1site.py --check`（档 3，
   它只要 stdlib + `by1check`，不需要 numpy）；
3. 用 `by1tiers.py` 复核档位归属（T2：模块级 import 与档位一致）。

## 三、M3 的具体做法

- 数据侧：`tools/by1site.py` 增加 `--graphs`，把 `graphs-golden.json` 里对应模型的
  `nodes`（`prim` / 形状）与边导出成 `site/data/<slug>.graph.js`；
- 视图侧：SVG 分层布局（按层分组、节点按 `prim` 上色），点节点显示它来自哪一层、
  以及该原语在两个发射器里的映射符号（复用 `by1attrs.emit_symbols` 的结果，导出时算好）；
- 判据：导出的节点数 == `graphs-golden.json` 里的节点数；golden 变了而站点没重导 -> `--check` 红。

## 四、不做

- 在线服务 / 账号 / 后端 API；
- 写回 `.by1` 或 IR（不做编辑器）；
- 三维/花哨可视化（能读、能查、能对账就够）。

## 五、怎么打开

```
双击 site/index.html
重新导出： python tools/by1site.py --out site/data
只核对：   python tools/by1site.py --check
```