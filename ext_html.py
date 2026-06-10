from __future__ import annotations

import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup

DEFAULT_INPUT = "/home/zihao/llm/paper_read/html/ClickPrompt.html"
DEFAULT_OUTPUT = "output/output_ClickPrompt.txt"


def clean_text(text: str) -> str:
    """清理多余空白"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def math_to_latex(math_tag) -> str:
    """将 <math> 转成 LaTeX 字符串，优先 annotation，其次 alttext"""
    ann = math_tag.find("annotation", {"encoding": "application/x-tex"})
    if ann and ann.get_text(strip=True):
        latex = ann.get_text(strip=True)
    elif math_tag.get("alttext"):
        latex = math_tag["alttext"].strip()
    else:
        latex = math_tag.get_text(" ", strip=True)

    latex = re.sub(r"\\displaystyle\s*", "", latex)
    return latex.strip()


def is_equation_table(classes: set[str]) -> bool:
    """判断是否为 arXiv 行间公式表格（单式或 equationgroup）。"""
    return "ltx_equationgroup" in classes or "ltx_equation" in classes


def extract_equation_table(table_tag) -> list[str]:
    """提取行间公式，每个 tr.ltx_equation 作为一条输出。"""
    lines = []
    for row in table_tag.select("tr.ltx_equation"):
        # 优先从 math annotation 取 LaTeX，避免编号与排版噪声
        math = row.find("math")
        if math is not None:
            text = math_to_latex(math)
        else:
            text = clean_text(row.get_text(" ", strip=True))
        if text:
            lines.append(text)
    return lines


def extract_authors(authors_div) -> str:
    """从 div.ltx_authors 提取作者姓名与单位。"""
    if not authors_div:
        return ""

    authors: list[str] = []
    for creator in authors_div.select("span.ltx_creator.ltx_role_author"):
        name_tag = creator.select_one("span.ltx_personname")
        if not name_tag:
            continue
        name = clean_text(name_tag.get_text())
        affs = [
            clean_text(a.get_text())
            for a in creator.select("span.ltx_affiliation_institution")
        ]
        affs = list(dict.fromkeys(a for a in affs if a))
        if affs:
            authors.append(f"{name} ({', '.join(affs)})")
        else:
            authors.append(name)
    return ", ".join(authors)


def extract_metadata(soup: BeautifulSoup, root) -> list[str]:
    """提取作者、发表日期、arXiv 信息与关键词，返回元数据行列表。"""
    meta: list[str] = []

    authors = extract_authors(root.select_one("div.ltx_authors"))
    if authors:
        meta.append(f"Authors: {authors}")

    published_parts: list[str] = []
    dates_tag = root.select_one("div.ltx_dates")
    if dates_tag:
        dates = clean_text(dates_tag.get_text())
        if dates:
            published_parts.append(dates)

    watermark = soup.select_one("#watermark-tr")
    if watermark:
        arxiv = clean_text(watermark.get_text())
        if arxiv and arxiv not in published_parts:
            published_parts.append(arxiv)

    if published_parts:
        meta.append(f"Published: {' / '.join(published_parts)}")

    keywords_tag = root.select_one("div.ltx_keywords")
    if keywords_tag:
        keywords = clean_text(keywords_tag.get_text())
        if keywords:
            meta.append(f"Keywords: {keywords}")

    return meta


def should_skip_paragraph(text: str) -> bool:
    """跳过 arXiv HTML 中无意义的占位段落。"""
    return text.lower() in {"by", ""}


def extract_body_blocks(root) -> list[str]:
    """按文档顺序提取标题、正文段落与行间公式。"""
    wanted_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "table"}
    blocks: list[str] = []
    seen: set[str] = set()

    for el in root.descendants:
        if not getattr(el, "name", None):
            continue
        if el.name not in wanted_tags:
            continue

        classes = set(el.get("class", []))

        if el.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            if "ltx_title" not in classes:
                continue
            text = clean_text(el.get_text(" ", strip=True))
            if text and text not in seen:
                blocks.append(text)
                seen.add(text)

        elif el.name == "p":
            if "ltx_p" not in classes:
                continue
            text = clean_text(el.get_text(" ", strip=True))
            if should_skip_paragraph(text) or text in seen:
                continue
            blocks.append(text)
            seen.add(text)

        elif el.name == "table":
            if not is_equation_table(classes):
                continue
            for eq in extract_equation_table(el):
                if eq and eq not in seen:
                    blocks.append(eq)
                    seen.add(eq)

    return blocks


def extract_paper(html: str) -> list[str]:
    """从 arXiv HTML 提取元数据与正文。"""
    soup = BeautifulSoup(html, "html.parser")

    root = soup.select_one("article.ltx_document")
    if root is None:
        root = soup.body if soup.body else soup

    metadata = extract_metadata(soup, root)

    for tag in root.select(
        ",".join([
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "button",
            "div.ltx_authors",
            "div.ltx_dates",
            "div.ltx_keywords",
            "figure",
            "span.ltx_note",
            "div.ltx_role_affiliation",
            "span.ltx_contact",
        ])
    ):
        tag.decompose()

    for cite in root.find_all("cite"):
        cite.unwrap()

    for a in root.find_all("a"):
        a.unwrap()

    for math_tag in root.find_all("math"):
        latex = math_to_latex(math_tag)
        math_tag.replace_with(f" {latex} ")

    body = extract_body_blocks(root)
    return metadata + body


def main() -> None:
    parser = argparse.ArgumentParser(description="从 arXiv HTML 提取论文文本")
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT,
        help=f"输入 HTML 文件路径（默认: {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"输出文本文件路径（默认: {DEFAULT_OUTPUT}）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    html = input_path.read_text(encoding="utf-8")
    blocks = extract_paper(html)

    output_path.write_text("\n\n".join(blocks), encoding="utf-8")
    print(f"提取完成 -> {output_path}（共 {len(blocks)} 块）")


if __name__ == "__main__":
    main()
