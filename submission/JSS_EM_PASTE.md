# JSS EM 填表用文本（逐字粘贴，不要改写）

来源：`paper/main.tex` / `paper/main.pdf`（30 页，大修版）。本文件只是「粘贴源」，
任何改动都必须先改 `main.tex`，再重新编译并跑 `tools/verify_paper_numbers.py`
（当前 305/305 通过）。旧稿号 `JSSOFTWARE-D-26-02395` 必须**先撤回**再投新版。

> **投稿状态（2026-09-19）**：大修版已投出，稿号 `JSSOFTWARE-D-26-02403`，
> EM 状态 `Submitted to Journal`。旧稿 `JSSOFTWARE-D-26-02395` 的撤回请求已于
> 2026-09-18 发出，编辑部尚未确认；已另发一封信把「新稿替代旧稿」的关系写清，
> cover letter 里也有 *Relationship to our earlier JSS submission* 一节。
> 因此下面的「第一步：撤回旧稿」只作为留档，不需要重复执行。

---

## 第一步：撤回旧稿（EM 里 Author 没有 withdraw 按钮，只能发信）

路径：Author Main Menu → Submissions Being Processed → 该稿件那一行的
**Send E-mail** → 收件人选 Editorial Office / Editor-in-Chief。
（该链接是弹窗页，必须在正常交互的浏览器里点开，脚本化访问会被 EM 拦成
"Security Warning: You have arrived at this page in an unauthorized manner."）

Subject：

```
Request to withdraw JSSOFTWARE-D-26-02395
```

正文：

```
Dear Editors,

I am writing to request the withdrawal of manuscript JSSOFTWARE-D-26-02395,
"Silent order corruption in per-sample evaluation artefacts: why the standard
negative control cannot detect it, and two tests that can".

An internal audit of our own artefacts, carried out after submission, showed that
two claims in the submitted version are stronger than our evidence supports. The
autocorrelation order test was reported as a detector of order corruption, whereas
its rejection region certifies residual order structure; and the cross-arm
agreement test was presented without the condition it needs, namely that the arms
carry sufficiently distinct loader orders, which most of our real defective groups
violate. Correcting these points changes the title, the abstract, two tables and
the claimed contribution, so a revision in place would not be honest to the
reviewers who received the current file.

I have rewritten the manuscript around what the artefacts do support, and I would
like to submit it as a new submission rather than ask the reviewers to work from
the withdrawn version. I apologise for the extra handling this causes, and I am
grateful for the time already spent on the paper.

Sincerely,
Mianhan Liu
Corresponding author
```

等编辑部确认状态变为 Withdrawn 之后再走第二步。

## 第二步：以新稿提交（Submit New Manuscript）

### Article Type

Research Paper（不选任何 Special Issue）。

### Title

```
Silent order corruption in per-sample evaluation artefacts: control degeneration, diagnostic blind spots, and a provenance contract
```

### Abstract（245 词）

```
Per-sample error tables, difficulty analyses, sample-level ablations and per-instance routing join artefacts written by different parts of a pipeline, and they join them positionally. In a widely used forecasting library one default, validation shuffles and test does not, permutes a per-window artefact taken from the validation split, and the fault survives the usual guards: aggregate metrics and model selection are invariant and the vectors are equally long. Our central result is structural: under uniform shuffling the permuted table is equal in distribution to the table produced by the shuffled-feature negative control for every order-symmetric statistic, so a null validated by that control is as consistent with a broken join as with a true null. When the label is derived from several independently written vectors, the control preserves the label law and no longer bounds the damage. Two diagnostics recover discarded order, an autocorrelation order test with exact randomisation moments and a cross-arm agreement test; on 356 runs we map their blind spots: the first flags 90.63% of arms in the real defect yet at most 1.67% of injected block and partial permutations short of full randomisation, and the second is defeated when two arms share a loader order, as 20 of 24 real defective groups do. The primary defence is write-side: a provenance contract that binds each array to an ordered sample-identifier digest and fails closed stops 20 of 25 injected identity faults; we enumerate the 5 it misses. Contract, diagnostics and artefacts are released.
```

### Keywords（6 个）

```
reproducibility
evaluation pipelines
silent faults
data provenance
negative controls
time series forecasting
```

### Highlights（5 条，最长 76 字符）

```
Shuffled evaluation loaders silently permute per-sample error artefacts
Every order-symmetric check passes: a mean cannot see a permutation
The shuffled-feature negative control is equal in law to the defect
Two order diagnostics, with the block and shared-order faults they miss
A sidecar contract on ordered sample identity stops 20 of 25 identity faults
```

### Authors（顺序不能变）

| # | 姓名 | 角色 | ORCID | 单位 / 地区 |
|---|---|---|---|---|
| 1 | Mianhan Liu | **Corresponding author** | 0009-0000-0274-1062 | Independent Researcher, Shanghai, China |
| 2 | Chen Chen | Co-author | 0009-0001-1991-9370 | Independent Researcher, Shanghai, China |

邮箱从 `main.tex` 的 `\correspemail` / `\secondemail` 取，不要手打。

### Funding

None（正文 Funding 一节同口径）。

### Publishing option

**Subscription**（0 费用）。不要选 Gold Open Access。

### Journal 自定义问题

- 是否在别处同时审稿：No
- 是否曾投过本刊：**Yes** —— 如实填写 `JSSOFTWARE-D-26-02395`，并说明该稿已由作者主动撤回，
  本次是按上述理由重写后的新版本。
- 是否有重叠/关联稿件：Yes —— 投往 Information Sciences 的 companion 稿件共用实验
  基础设施，但问题与结论不重叠；正文 "Relation to other work by the authors" 一节
  已列出共用项与各自独有贡献。
- 生成式 AI 使用：正文 "Declaration of generative AI" 一节已披露。

### 上传清单

| Item Type | 文件 |
|---|---|
| Manuscript（LaTeX 路线） | `submission/OrderProvenance_EM_latex_20260918.zip`（单文件自包含 `main.tex`，无子目录） |
| Declaration of Interest（强制） | `submission/declaration_of_interest_elsevier.docx` |
| Highlights | `submission/highlights.docx` |
| Cover Letter | `submission/cover_letter.docx` |
| Author Statement / CRediT | `submission/credit_author_statement.md` |

> EM 只把 manuscript item 放进编译目录，其余 "LaTeX Source Files" 只当附件，所以
> `main.tex` 必须单文件自包含。用 `tools/make_em_latex_flat.py` 生成。
