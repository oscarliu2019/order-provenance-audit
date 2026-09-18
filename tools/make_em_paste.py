#!/usr/bin/env python3
"""Render the Editorial Manager paste pack from the manuscript source."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "main.tex"
NUMBERS = ROOT / "paper" / "numbers.tex"
HIGHLIGHTS = ROOT / "submission" / "highlights.md"
OUT = ROOT / "submission" / "JSS_EM_PASTE.md"

OLD_MS = "JSSOFTWARE-D-26-02395"


def macros() -> dict[str, str]:
    pat = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
    return dict(pat.findall(NUMBERS.read_text(encoding="utf-8")))


def block(name: str) -> str:
    text = MAIN.read_text(encoding="utf-8")
    m = re.search(rf"\\begin\{{{name}\}}(.*?)\\end\{{{name}\}}", text, re.S)
    if not m:
        raise SystemExit(f"{name} block not found in main.tex")
    return m.group(1).strip()


def detex(s: str, nums: dict[str, str]) -> str:
    s = re.sub(r"(?<!\\)%.*?$", "", s, flags=re.M)
    s = re.sub(r"\\([A-Za-z]+)\{\}", lambda m: nums.get(m.group(1), m.group(0)), s)
    s = re.sub(r"\\(?:emph|textit|textbf)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\([A-Za-z]+)\b", lambda m: nums.get(m.group(1), m.group(0)), s)
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_")
    s = s.replace("~", " ").replace("``", '"').replace("''", '"')
    s = re.sub(r"\s+", " ", s).strip()
    if "\\" in s:
        raise SystemExit(f"unresolved LaTeX in rendered text: {s[:200]}")
    return s


def title(nums: dict[str, str]) -> str:
    text = MAIN.read_text(encoding="utf-8")
    m = re.search(r"\\title\[mode = title\]\{(.*?)\n?\}", text, re.S)
    if not m:
        raise SystemExit("title not found")
    return detex(m.group(1), nums)


def keywords(nums: dict[str, str]) -> list[str]:
    return [detex(k, nums) for k in block("keywords").split(r"\sep")]


def highlights() -> list[str]:
    bullets = [
        line[2:].strip()
        for line in HIGHLIGHTS.read_text(encoding="utf-8").splitlines()
        if line.startswith("- ")
    ]
    if not 3 <= len(bullets) <= 5:
        raise SystemExit(f"Elsevier wants 3-5 highlights, found {len(bullets)}")
    for b in bullets:
        if len(b) > 85:
            raise SystemExit(f"highlight over 85 chars ({len(b)}): {b}")
    return bullets


def main() -> None:
    nums = macros()
    abstract = detex(block("abstract"), nums)
    words = len(abstract.split())
    kw = keywords(nums)
    hl = highlights()
    pages = "30"

    body = f"""# JSS EM 填表用文本（逐字粘贴，不要改写）

来源：`paper/main.tex` / `paper/main.pdf`（{pages} 页，大修版）。本文件只是「粘贴源」，
任何改动都必须先改 `main.tex`，再重新编译并跑 `tools/verify_paper_numbers.py`
（当前 305/305 通过）。旧稿号 `{OLD_MS}` 必须**先撤回**再投新版。

---

## 第一步：撤回旧稿（EM 里 Author 没有 withdraw 按钮，只能发信）

路径：Author Main Menu → Submissions Being Processed → 该稿件那一行的
**Send E-mail** → 收件人选 Editorial Office / Editor-in-Chief。
（该链接是弹窗页，必须在正常交互的浏览器里点开，脚本化访问会被 EM 拦成
"Security Warning: You have arrived at this page in an unauthorized manner."）

Subject：

```
Request to withdraw {OLD_MS}
```

正文：

```
Dear Editors,

I am writing to request the withdrawal of manuscript {OLD_MS},
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
{title(nums)}
```

### Abstract（{words} 词）

```
{abstract}
```

### Keywords（{len(kw)} 个）

```
{chr(10).join(kw)}
```

### Highlights（{len(hl)} 条，最长 {max(len(b) for b in hl)} 字符）

```
{chr(10).join(hl)}
```

### Authors（顺序不能变）

| # | 姓名 | 角色 | ORCID | 单位 / 地区 |
|---|---|---|---|---|
| 1 | Mianhan Liu | **Corresponding author** | 0009-0000-0274-1062 | Independent Researcher, Shanghai, China |
| 2 | Chen Chen | Co-author | 0009-0001-1991-9370 | Independent Researcher, Shanghai, China |

邮箱从 `main.tex` 的 `\\correspemail` / `\\secondemail` 取，不要手打。

### Funding

None（正文 Funding 一节同口径）。

### Publishing option

**Subscription**（0 费用）。不要选 Gold Open Access。

### Journal 自定义问题

- 是否在别处同时审稿：No
- 是否曾投过本刊：**Yes** —— 如实填写 `{OLD_MS}`，并说明该稿已由作者主动撤回，
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
"""
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: abstract {words} words, "
          f"{len(kw)} keywords, {len(hl)} highlights")


if __name__ == "__main__":
    main()
