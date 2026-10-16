# 外部复现 #1 —— Colab CPU quickstart（John6666）

- **来源**: https://discuss.huggingface.co/t/by1-describing-and-validating-llm-architectures/190467/2
- **作者**: John6666（HuggingFace 论坛，by1 帖子 2 楼）
- **日期**: 2026-10-16
- **钉住的版本**: ffaf2a4cca4c95296fccc4c81aa36f3b79fe070a
- **环境**: hosted Linux / Colab，Python 3.13.16，无 GPU / PyTorch / Transformers

**结论摘要**：CPU 入门流程复现成功（4 层 1,049,728 参数、2 层 557,696、
输出 (1, 8, 256) 全部与 README 一致），并独立手推核对了两个参数量。发现一处
check/run 边界缺口：hello `head_dim=31` check 过、run 在 NumPy 广播里裸崩 ——
2026-10-16 当天修复（by1ir.validate「旋转宽度必须是偶数」+ by1mutate 反例 5 个
+ 近旁正例 2 个，判据按旋转宽度、不全局禁奇数 —— 全部照发现者的设计意见）。
另记三件：venv/ensurepip 在该 Colab 镜像失败（非 by1 缺陷，备选路线已进
README）；README 版本串过期（0.9.0/1.0 vs 真相源 1.0.0-rc1/1.1，已修）；
中文 CLI 对非中文用户是障碍（待办：语言中立状态码）。

他的四层证据表、"第二份规格"警告与 use-case 分析见原文末几节 —— 对产品定义
的贡献，后续吸收时署名引用。

以下为报告原文（HTML 转文本，内容未改动）：

---
For now, I tried running it on Colab:

CPU quickstart: what worked

I tried the Hello CPU example at the pinned by1 commit ffaf2a4. The main result matched the README quickstart:

CPU case
check
NumPy run
Parameters
Output shape

Original 4 * Attention(...)
Passed
Passed
1,049,728
(1, 8, 256)

Changed to 2 * Attention(...)
Passed
Passed
557,696
(1, 8, 256)

Changing that one declaration changed the number of decoder blocks and the allocated parameters, while the FFNs followed the new layer count and the output shape stayed the same. I reproduced the central cases first through the source entry point, and again through an installed by1 command in a separate Colab CPU run. No GPU, model weights, or Transformers installation was involved.

One installation caveat: the exact README virtual-environment sequence did not finish on this Colab image. The ordinary python3 -m venv command stopped during ensurepip, before by1 installation. I then used venv --without-pip and the base interpreter’s pip --python option. That alternative installed by1 successfully, and its console command passed the tests above. This is evidence of an installed-CLI quickstart through an alternative route, not evidence of a by1 packaging defect or of universal Colab installation behavior. The precise environment and commands are below.

Would this kind of architecture description be useful?

Yes, potentially, especially as an inspectable and machine-checkable contract between a model’s description, its checkpoint, and a downstream implementation. Its value seems less about replacing an established inference runtime and more about making assumptions and validation boundaries explicit enough for another person to review.

Your product definition names three audiences. All three make sense as use-case hypotheses:

Checkpoint publishers and converters: compare the stated architecture with config fields, physical tensor names and shapes, tied weights, and export rules.

Architecture reviewers: identify meaningful differences in attention, normalization, biases, positional behavior, or auxiliary components without reconstructing the whole design from multiple implementation files.

Second-backend implementers: use a declaration and a checkable report to understand what the new implementation must preserve, without expecting by1 itself to supply every native kernel.

This is complementary to existing work. The llama.cpp guide to adding an architecture describes conversion, tensor mappings, architecture registration, and graph construction as real tasks; Transformers’ WeightConverter already addresses weight-format transformations. A useful distinction for by1 would be a single reviewable statement of the model’s intended structure, accompanied by evidence showing which claims were checked. I haven’t tried a real checkpoint conversion with by1 or measured engineering time saved, so I mean this as a concrete possible use, not a demonstrated productivity gain.

A small check versus run observation

I also varied one Hello field at a time. head_dim=32 and head_dim=30 both passed check and completed run. With head_dim=31, however, check passed but run failed with a NumPy shape/broadcast error in the interleaved RoPE path. That distinction reproduced with the installed CLI. In the source-entry probe, kv=0 also passed the structural check but was explicitly reported as unsupported during execution; a nonexistent mechanism name was detected by check. These are different outcomes, not evidence of one general failure.

There are at least two reasonable ways to handle this, depending on what check is meant to promise:

If check should establish executability for a specified mechanism/backend, a cheap conditional preflight could reject this unsupported rotary pairing with a diagnostic pointing to head_dim or the active rotary dimension.

If check intentionally covers structural and arithmetic invariants only, the separation is defensible. A short quickstart note distinguishing a successful static check from executable operator combinations would prevent a clean check from being mistaken for a full forward-pass guarantee.

The observation is specific to this Hello configuration and RoPE implementation; it is not a claim that every model requires an even full attention-head dimension. Your testing guide already takes care to separate passes, failures, and skips, so this would be a small extension of that clarity rather than a change of direction.

My practical takeaway: the Hello demonstration did what it promised at the CPU execution level. The most useful next proof of value would be one compact declaration plus check/report example tied to a real conversion or architecture review; the exact next step could depend on which audience finds that most useful. Reproduction details and optional branches follow.

Reproduction: exact source, environments, commands, and results
What was tested

Both Colab probes used Linux CPU and Python 3.13.16, with by1 checked out at commit ffaf2a4cca4c95296fccc4c81aa36f3b79fe070a. The fixed revision makes the observations reproducible even if main changes later. The first probe invoked src/lang/by1.py directly, with NumPy 2.1.3. The second installed the project in a separate pip-free venv, then invoked its console script, with NumPy 2.5.4 inside that environment and by1 1.0.0-rc1 / by1-ir 1.1. Neither run used a GPU, PyTorch, Transformers, pretrained weights, a model download, or a tokenizer.

The Hello declaration tested was:

model Hello {
  hparams { d_model = 128, vocab = 256 }
  stack main {
    pattern = 4 * Attention(q = 4, kv = 2, head_dim = 32)
    main[:] >> FFN(hidden = 512)
  }
}

The second successful variant changed only the 4 on the pattern line to 2. Vocabulary size, sequence length, head dimensions, and FFN width stayed fixed. From the repository root, the installed-CLI commands are equivalent to:

alternate_venv/bin/by1 check models/hello.by1
alternate_venv/bin/by1 run models/hello.by1 --seq 8
# Edit pattern = 4 * Attention(...) to pattern = 2 * Attention(...)
alternate_venv/bin/by1 check models/hello.by1
alternate_venv/bin/by1 run models/hello.by1 --seq 8

The probe saved each variant separately, preserving the original declaration. Installation commands follow below.

Observed cases

Declaration or change
check
run
What the output established

Unmodified Hello, four layers
exit 0
exit 0
1,049,728 parameters; (1, 8, 256)

Two layers
exit 0
exit 0
557,696 parameters; (1, 8, 256)

head_dim=30, four layers
exit 0
exit 0
1,037,440 parameters; forward completed

head_dim=31, four layers
exit 0
exit 1
1,043,584 reported before forward failed

kv=0
exit 0
unsupported
Explicit non-execution result

Unknown mechanism name
nonzero
nonzero
Invalid reference caught before normal forward

What this can and cannot establish

A successful run shows that the NumPy path executed these inputs and reported the expected parameter allocation and shape. It does not independently validate numerical semantics, checkpoint loading, cache behavior during generation, quantized kernels, or deployment performance. Those require different experiments.

Installation on Colab: the standard venv failure and a working alternative
Separate platform setup from package installation

In both Colab runs, the standard virtual-environment creation stage failed when Python invoked ensurepip --upgrade --default-pip. The process returned a nonzero status before any attempt to install the by1 package. The captured log did not establish the low-level cause of that ensurepip failure. It should not be attributed to by1 or generalized to other Colab images or desktop Linux installations.

In the later probe, I successfully installed by1 by creating a venv without pip bootstrapping, then controlling its interpreter with the base Python’s pip. This procedure is documented in pip’s Managing a different Python interpreter guide; Python documents the relevant options in venv and ensurepip.

From the repository root, the tested fallback was:

python3 -m venv --without-pip alternate_venv
python3 -m pip --python alternate_venv install -e '.[run]'
alternate_venv/bin/by1 --version
alternate_venv/bin/by1 check models/hello.by1
alternate_venv/bin/by1 run models/hello.by1 --seq 8

Each step completed successfully in the returned Colab log. This was a genuine editable installation and console-script test, separate from the earlier direct-source run. The reported CLI version was by1 1.0.0-rc1 with IR 1.1. A subsequent layer-count edit also worked through that executable.

One environment detail matters: alternate_venv/bin/python -m pip show by1 numpy still failed with No module named pip. The base interpreter’s pip had managed the target venv, so editable installation and CLI execution succeeded, but the target venv did not have its own invocable pip module.

How I would present this to new users

The README’s ordinary instructions remain the default wherever venv creation works. The alternative is relevant specifically to a failure in venv/ensurepip, before project installation. A dependency-resolution or package-build failure later in the process would need separate diagnosis.

I did not test Windows or Python 3.9; these results are from hosted Linux on Python 3.13.16.

The `head_dim` control: mechanism, failure categories, and two design paths
A small differential test

The baseline declaration is Attention(q=4, kv=2, head_dim=32). Replacing only 32 with 30 left both commands successful. Replacing it with 31 left the static check successful but caused a NumPy broadcasting failure during run. The logged incompatible shapes included (1, 4, 8, 16) and (8, 15). The 1,043,584 parameter figure for that negative case was printed before the failed forward pass; it is not a successful run or a validated output count.

This is consistent with the pinned NumPy executor: the tested interleaved RoPE path separates alternating feature components and uses sine/cosine tables. An odd full rotary width yields unequal alternating groups for this implementation. This is a code-supported explanation of this mismatch, not a statement about all RoPE variants. Partial rotary embeddings and other layouts may impose a constraint on a rotary sub-dimension rather than the entire attention head width.

Two nearby observations keep the conclusion narrow:

With head_dim=30, execution succeeded. Thus this is not simply a hard-coded requirement that Hello use exactly 32.

With kv=0, a separate source-entry test reached an explicit unsupported result in the execution stage. That is different from an uncaught broadcasting exception. A nonexistent mechanism was caught by check instead. These show different boundaries of name validity, unsupported combinations, and runtime-shape validity.

This was a feature-dimension control, not a trained-model or weight-corruption test. Its value is that one number distinguishes two validation stages.

Two compatible API designs

Path A: check promises backend-specific executability. If that is the intended contract, then a cheap validation rule before the forward pass could reject the nonrepresentable rotary pairing. It need not be a general rule that all head_dim values must be even. It could be conditional on the active RoPE mechanism, its selected rotary dimension, the pairing convention, and the chosen backend. A useful diagnostic might say that the requested rotary width cannot be paired under this configuration and identify the relevant declared field.

Path B: check promises only static structural invariants. If that is the intended contract, the fact that a structurally valid declaration can be unexecutable in one backend may be by design. An explicit separation such as structure validated / operator combination unsupported / forward not attempted is then easier to reason about than silently broadening the meaning of check. The first-use guide could state what the green check does and does not guarantee; optionally, a future execution preflight could be a distinct command or mode rather than added to every static check.

A minimal regression fixture would include 32 as the baseline positive, 30 as a nearby positive, and 31 as the negative for this interleaved RoPE case. Depending on the chosen semantics, the expected negative is either a static diagnostic or an explicit backend-incompatibility report before NumPy broadcasting occurs. It should not be made a global test that rejects every odd attention width regardless of the selected position scheme.

The result offers a low-cost, reproducible choice about validation boundaries, without requiring a redesign.

Use case 1: publishing or converting a checkpoint
For a publisher or converter, the relevant questions include: Are the declared dimensions consistent with the stored tensors? Are Q, K, and V projections fused or separate? Do some projection layers have biases while others do not? Is the output head tied to the input embeddings or independently stored? Are normalization conventions and position parameters represented in the description, rather than silently assumed from model family? How are auxiliary, multimodal, or expert tensors organized? What conversion-time permutations or reshaping are required?

This is adjacent to, not a replacement for, two substantial existing systems:

In the llama.cpp model-addition guide, conversion to GGUF, metadata/tensor mapping, architecture registration, and the GGML forward graph are separate engineering steps. Some modifications are deliberately done at conversion time rather than as graph operators. A specification may help relate those decisions, but it does not eliminate them.

Transformers’ WeightConverter has composable operations for renaming, splitting and concatenating projections, transposing layouts, handling experts, and more. It already addresses difficult weight-format translation. By1 need not duplicate this functionality to be useful as a higher-level statement of architecture or validation conditions.

Parameter sharing deserves explicit treatment. The safetensors discussion of shared tensors explains why storage and sharing semantics need special treatment; a file’s tensor names alone are not always sufficient to reconstruct aliasing intentions. An explicit statement of whether a head and embedding table should be tied would be useful, provided it is verified at the appropriate layer rather than merely declared.

I have not tested a real checkpoint conversion with by1. A reviewable, single-model example showing how the declaration and report help alongside the existing converter would make the use case easier to judge, without claiming unmeasured time savings.

Use case 2: reviewing an architecture, and when a DSL adds little
A reviewer may want to answer what is structurally claimed, which claims were checked against published artifacts, and which checks actually executed operators. If those answers can be recovered from a .by1 definition and its report without reading several source files, that seems a genuine usability benefit. The repo’s complete-model examples already indicate that such distinctions, including parameter tying, are important to the design.

One possible review demonstration is to compare two related declarations differing in one meaningful property. Examples could be toggling an output-head tie, changing the norm type, or changing a permitted bias configuration, assuming the current language supports the selected change. The value would be that the difference is legible in the declaration and reflected in a report or a negative control.

The case against adding another layer of tooling is also worth acknowledging. If someone only wants to run a common, already-supported model, the existing configuration class, implementation, and runtime may be enough. If by1 adds a declaration that no one reviews, validates, or maintains alongside the real implementation, it risks becoming another specification to keep synchronized. That is not an argument against the project. It is a condition for usefulness: the declaration should produce a report, artifact, or check someone already wants, rather than being maintained only for its own sake.

Another useful distinction is which artifact is authoritative. For some projects, the checkpoint/config is authoritative, and the architecture declaration is derived and checked. For others, the declaration is the design input used to construct an implementation or export plan. Both may be valuable, but mixing them without stating which source wins during a discrepancy can confuse reviewers. By1 already has mechanisms to draft descriptions from artifacts and verify them; those are worth treating as complementary workflows rather than forcing one universal direction.

The most persuasive review demo may therefore be something like declaration, manifest, discrepancy, resolved interpretation, with enough evidence to reproduce the conclusion. Marking unverified or unsupported fields can matter as much as displaying passing checks. The author is already explicit about this separation in TESTING.md.

Use case 3: a second backend, and why validation has multiple layers
A practical distinction is between four increasingly strong kinds of evidence:

Level
Example evidence
What it still cannot guarantee

Structural specification
Fields, mechanism names, layers, shapes, dtypes, ties
Correct arithmetic or execution

Checkpoint contract
Every required tensor is accounted for and loaded or explicitly rejected
Correct use of those tensors

Forward behavior
A fixed input and weights produce logits close to a trustworthy reference
Long-running generation and state equivalence

Deployment behavior
The exported artifact runs in an unmodified target runtime
All architectures, dtypes, hardware paths, or future versions

The repository already states the limits of its evidence. It distinguishes testing tiers, and the criteria-review document asks whether the pass criteria, tolerance, coverage definitions, and chosen sample models are appropriate. A pass under documented criteria is valuable, but it does not itself independently justify those criteria. I would view an architecture DSL as especially valuable if it makes the proof obligations visible and allows an independent test to falsify the wrong declaration.

If there is an independent target backend, a useful first demonstration could be a small model and one fixed prompt with three separate reports: declaration/weight contract, prefill logits parity, and a few incremental decode steps. If the goal is merely to validate the language’s expressiveness, the first report may be enough; if the claim is “this is a functioning deployed model,” the latter reports matter more. Where hardware or checkpoint availability prevents a test, SKIP or “not checked” is more informative than silently counting it as a pass.

The public GPU validation results describe comparisons on real checkpoints, with an important limitation: the compared routes can share Transformers operators. I read that as the author’s reported correctness evidence under those operators, not an independent test of a native runtime or a performance result. My Colab Hello test is a different, much smaller piece of evidence and should not be added to the model-parity numbers as though it verified them.

As far as I can tell, your stated upstream-first direction for llama.cpp is compatible with this scope: use existing, supported runtime capabilities for deployment, and focus by1 on architecture/checkpoint contracts and checking. A native adapter would be a separate deliverable with its own tests, not a prerequisite for a useful review contract. That is a design boundary worth preserving.

Independent arithmetic cross-check: why the two parameter counts make sense
Let the model width be 128, vocabulary 256, FFN hidden width 512, query heads 4, key/value heads 2, and head dimension 32. A full query projection maps 128 to 4 x 32 = 128, so it contributes 128 x 128 = 16,384 weights. The K and V projections each map 128 to 2 x 32 = 64, contributing 8,192 weights each. Attention output projection maps 128 back to 128, contributing 16,384. Thus the attention projections contribute 49,152 weights per layer, under the no-bias configuration used in the small example.

For the gated FFN, the up, gate, and down projections contribute three matrices of 128 x 512, or 196,608 weights per layer. Two RMSNorm vectors add 256 parameters per layer. The block subtotal is therefore:

attention                 49,152
FFN                      196,608
two norms                    256
--------------------------------
per decoder layer        246,016

The global parameters include the 256 x 128 = 32,768 embedding matrix, a separate output head of 32,768 weights, and a 128-parameter final normalization, totaling 65,664. Therefore:

4 x 246,016 + 65,664 = 1,049,728
2 x 246,016 + 65,664 =   557,696
Difference              =   492,032

These numbers depend on the model’s declared untied head, normalization, head sizes, gated FFN, and lack of projection biases. Changing tying, biases, experts, or dimensions changes the formula. Most importantly, matching a parameter count is necessary but nowhere near sufficient to show that the network computes the intended function. Different graphs can use the same number of parameters.

What a useful next demonstration would look like, depending on the audience
The next useful example depends on the reader’s purpose; one Show and Tell post does not need a broad benchmark suite to establish value.

Reader’s actual goal
Lowest-cost useful evidence
When to go deeper

“Can I try the language?”
Hello check, run, and four-layer/two-layer comparison
Only if a first-run failure or confusing step appears

“Can I audit this checkpoint?”
Config and tensor-manifest comparison, with explicit ties and gaps
If the mismatch concerns loading or computation

“Will it help a converter?”
One real conversion’s assumptions shown in the declaration and report
If comparing engineering effort against current workflow

“Can I implement another backend?”
Explicit operator/tensor contract and one negative control
Then fixed-weight logits and incremental state tests

“Can I deploy this model?”
Confirmation that the actual exported artifact loads and runs upstream
Then quantitative fidelity and platform coverage

For the reviewer path, a deliberately changed architectural property is a useful control. A case such as tied versus untied head or a supported norm variant can demonstrate whether the system distinguishes semantically different declarations, not just differently written versions of the same dimensions. A reviewer can compare the declaration with the reported tensor or execution consequences. The selected property should be one that the current by1 language actually supports; I am not proposing new syntax for the sake of the demonstration.

For a second backend, “the generated graph runs” and “the backend implements the same function” remain separate claims. The strongest low-cost next check would usually fix random seeds and exact weights, feed a short input to two sufficiently independent paths, compare logits under a justified tolerance, and report what code those paths share. If the paths share the relevant operator implementation, that supports configuration/wiring parity but is weaker as an independent arithmetic oracle. A later test could change sequence length and compare incremental decode with cache enabled; none of this is required to answer the present first-use question.

For this first-use question, reproducing Hello, clarifying the checker/executor boundary, and identifying one reviewable workflow would already be meaningful progress. Additional research can then follow whichever audience actually shows interest. That keeps the scope proportional to the evidence.

Small first-use and documentation details, separate from the core result
A few minor things are worth recording, but I would rank them below the successful CPU demonstration and the possible contract/review use case.

Output language. The English README translates the expected Chinese CLI messages for the Hello example, so the main success condition was identifiable. A reader who does not know Chinese may still struggle with later help and diagnostics, particularly the difference between a failure and an unsupported case. Possible low-cost options are a few additional translated example messages, language-neutral status codes, or an optional English output mode. The current Chinese output is not itself a correctness issue.

Version label. At the pinned commit, the README’s first code block says by1 0.9.0 / by1-ir 1.0, while the installed CLI reports 1.0.0-rc1 / by1-ir 1.1. That is a specific documentation freshness mismatch at the tested revision, not an indication that the runtime is using the wrong code. A small README synchronization would reduce ambiguity for bug reports. Since the repository may change, the statement should always be tied to the revision used for this experiment.

First-use experience versus a synthetic test. My reported results come from deliberate probes and logs. I did not conduct an observational usability study with several beginners, nor do I want to pretend I felt confused at a step I could explain from the documentation. These are potential confusion points, not observations from a broader usability study.

The short version is that I could reproduce the CPU demo, and I can see a credible role for a declarative specification in checkpoint and architecture review. The head_dim=31 case seems like a useful small example for documenting where static checking ends and backend execution begins.


---

（存档说明：本文件由论坛帖 2 楼原文提取，2026-10-16。）
