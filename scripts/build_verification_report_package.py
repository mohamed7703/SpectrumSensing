"""Builds comprehensive verification download packages:
1. HTML report with MathJax and embedded/relative plot views.
2. Complete ZIP bundle containing MD, HTML, all 19 plots, and all 19 logs.
3. Copies all download artifacts to C:\\Users\\aamah\\Downloads for instant access.
"""

import os
import shutil
import zipfile
from pathlib import Path

WORKSPACE = Path(r"s:\Projects\SpectrumSensing")
DOCS_DIR = WORKSPACE / "docs"
PLOTS_DIR = WORKSPACE / "results" / "plots"
LOGS_DIR = WORKSPACE / "results" / "logs"
DOWNLOADS_DIR = Path(r"C:\Users\aamah\Downloads")

MD_REPORT = DOCS_DIR / "COMPREHENSIVE_VERIFICATION_REPORT.md"
HTML_REPORT = DOCS_DIR / "COMPREHENSIVE_VERIFICATION_REPORT.html"
ZIP_PACKAGE = DOCS_DIR / "Spectrum_Sensing_Comprehensive_Verification_Package.zip"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cognitive Radio Spectrum Sensing — Comprehensive Verification Report</title>
  
  <!-- MathJax for rendering KaTeX / LaTeX formulas -->
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }
    };
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <!-- Marked for Markdown rendering -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

  <!-- Highlight.js for code syntax coloring -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

  <style>
    :root {
      --primary: #1e3a8a;
      --secondary: #0284c7;
      --success: #16a34a;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
    }
    
    * {
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.65;
      color: var(--text);
      background-color: var(--bg);
      margin: 0;
      padding: 0;
    }

    .header-banner {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0284c7 100%);
      color: white;
      padding: 48px 24px;
      text-align: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    .header-banner h1 {
      margin: 0 0 12px 0;
      font-size: 2.5rem;
      font-weight: 800;
      letter-spacing: -0.025em;
    }

    .header-banner p {
      font-size: 1.15rem;
      max-width: 800px;
      margin: 0 auto 20px auto;
      opacity: 0.9;
    }

    .badges {
      display: flex;
      gap: 10px;
      justify-content: center;
      flex-wrap: wrap;
      margin-top: 16px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      background: rgba(255,255,255,0.15);
      backdrop-filter: blur(8px);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.875rem;
      font-weight: 600;
      border: 1px solid rgba(255,255,255,0.25);
    }

    .container {
      max-width: 1140px;
      margin: -24px auto 60px auto;
      background: var(--card-bg);
      padding: 48px;
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.06);
      border: 1px solid var(--border);
    }

    @media (max-width: 768px) {
      .container {
        padding: 24px 16px;
        margin: 0 auto;
        border-radius: 0;
      }
      .header-banner h1 {
        font-size: 1.8rem;
      }
    }

    /* Markdown Styling */
    h1, h2, h3, h4, h5, h6 {
      color: #0f172a;
      font-weight: 700;
      margin-top: 36px;
      margin-bottom: 16px;
      padding-bottom: 8px;
    }

    h1 {
      font-size: 2rem;
      border-bottom: 2px solid var(--border);
    }

    h2 {
      font-size: 1.5rem;
      border-bottom: 1px solid var(--border);
      color: var(--primary);
    }

    h3 {
      font-size: 1.25rem;
      color: #334155;
    }

    p, li {
      color: #334155;
      font-size: 1.025rem;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin: 24px 0;
      font-size: 0.95rem;
      background: white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      border-radius: 8px;
      overflow: hidden;
    }

    th, td {
      padding: 12px 16px;
      text-align: left;
      border: 1px solid var(--border);
    }

    th {
      background-color: #f1f5f9;
      font-weight: 600;
      color: #0f172a;
    }

    tr:nth-child(even) {
      background-color: #f8fafc;
    }

    tr:hover {
      background-color: #f1f5f9;
    }

    code {
      background-color: #f1f5f9;
      color: #be185d;
      padding: 3px 6px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.875em;
    }

    pre code {
      background-color: transparent;
      color: inherit;
      padding: 0;
    }

    pre {
      background-color: #0f172a;
      color: #f8fafc;
      padding: 20px;
      border-radius: 10px;
      overflow-x: auto;
      font-size: 0.9rem;
    }

    blockquote {
      border-left: 4px solid var(--secondary);
      margin: 20px 0;
      padding: 12px 20px;
      background-color: #f0f9ff;
      border-radius: 0 8px 8px 0;
      color: #0369a1;
    }

    img.plot-preview {
      display: block;
      max-width: 100%;
      height: auto;
      margin: 20px auto 30px auto;
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.1);
      border: 1px solid var(--border);
      transition: transform 0.2s ease;
    }

    img.plot-preview:hover {
      transform: scale(1.01);
    }

    .toc {
      background-color: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 28px;
      margin: 28px 0;
    }

    .toc h4 {
      margin-top: 0;
      color: var(--primary);
    }

    .toc ul {
      margin-bottom: 0;
      padding-left: 20px;
    }

    .toc a {
      color: var(--secondary);
      text-decoration: none;
    }

    .toc a:hover {
      text-decoration: underline;
    }

    .footer {
      text-align: center;
      margin: 40px 0;
      color: var(--text-muted);
      font-size: 0.875rem;
    }

    @media print {
      body {
        background-color: white;
      }
      .container {
        box-shadow: none;
        border: none;
        margin: 0;
        padding: 0;
        max-width: 100%;
      }
      .header-banner {
        background: white;
        color: black;
        border-bottom: 2px solid #000;
        padding: 20px 0;
      }
      .badges {
        display: none;
      }
    }
  </style>
</head>
<body>

  <div class="header-banner">
    <h1>Cognitive Radio Spectrum Sensing</h1>
    <p>Comprehensive Verification, Test Suite Audit & Empirical Benchmark Report</p>
    <div class="badges">
      <span class="badge">Python 3.10+</span>
      <span class="badge" style="background:#15803d;">205 / 205 Tests Passing (100%)</span>
      <span class="badge">Phases 1-4 Complete</span>
      <span class="badge">19 Experiment Logs</span>
      <span class="badge">19 Publication Plots</span>
    </div>
  </div>

  <div class="container">
    <div id="content">Loading verification report...</div>
    <div class="footer">
      AI-Driven Cognitive Radio Spectrum Sensing Framework &bull; Verified 2026 &bull; Antigravity Autonomous Coding Agent
    </div>
  </div>

  <script>
    // Embed the raw markdown content
    const rawMarkdown = `RAW_MARKDOWN_PLACEHOLDER`;

    // Process image references: convert markdown links like
    // [results/plots/xxx.png](file:///...) into responsive <img> previews
    const processedMarkdown = rawMarkdown.replace(
      /\\[`results\\/plots\\/([a-zA-Z0-9_\\-\\.]+)\\.png`\\]\\([^)]+\\)/g,
      '<img class="plot-preview" src="plots/$1.png" alt="$1 Verification Plot" loading="lazy" />\\n\\n[Download High-Res Figure (plots/$1.png)](plots/$1.png)'
    );

    // Configure marked
    marked.setOptions({
      gfm: true,
      breaks: false,
      highlight: function(code, lang) {
        const language = highlight.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
      }
    });

    document.getElementById('content').innerHTML = marked.parse(processedMarkdown);

    // Re-trigger MathJax typesetting after dynamic Markdown insertion
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise();
    }
  </script>
</body>
</html>
"""

def build_package():
    print(f"Reading markdown report from {MD_REPORT}...")
    with open(MD_REPORT, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Escape backticks and backslashes for JS template string
    escaped_md = md_content.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html_output = HTML_TEMPLATE.replace("RAW_MARKDOWN_PLACEHOLDER", escaped_md)

    print(f"Writing HTML report to {HTML_REPORT}...")
    with open(HTML_REPORT, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"Creating comprehensive ZIP package at {ZIP_PACKAGE}...")
    with zipfile.ZipFile(ZIP_PACKAGE, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add Markdown report
        zipf.write(MD_REPORT, arcname="COMPREHENSIVE_VERIFICATION_REPORT.md")
        # Add HTML report
        zipf.write(HTML_REPORT, arcname="COMPREHENSIVE_VERIFICATION_REPORT.html")

        # Add all 19 plots
        print("Packing 19 verification plots...")
        for plot_path in PLOTS_DIR.glob("*.png"):
            zipf.write(plot_path, arcname=f"plots/{plot_path.name}")

        # Add all 19 experiment logs
        print("Packing 19 experiment logs...")
        for log_path in LOGS_DIR.glob("*.md"):
            zipf.write(log_path, arcname=f"logs/{log_path.name}")

    print(f"ZIP package created: {ZIP_PACKAGE.stat().st_size / (1024*1024):.2f} MB")

    # Copy files to user's Downloads folder
    if DOWNLOADS_DIR.exists():
        print(f"Copying files to Downloads directory ({DOWNLOADS_DIR})...")
        dl_md = DOWNLOADS_DIR / "COMPREHENSIVE_VERIFICATION_REPORT.md"
        dl_html = DOWNLOADS_DIR / "COMPREHENSIVE_VERIFICATION_REPORT.html"
        dl_zip = DOWNLOADS_DIR / "Spectrum_Sensing_Comprehensive_Verification_Package.zip"

        shutil.copy2(MD_REPORT, dl_md)
        shutil.copy2(HTML_REPORT, dl_html)
        shutil.copy2(ZIP_PACKAGE, dl_zip)

        print(f"Copied to: {dl_md}")
        print(f"Copied to: {dl_html}")
        print(f"Copied to: {dl_zip}")
    else:
        print(f"Warning: {DOWNLOADS_DIR} does not exist.")

if __name__ == "__main__":
    build_package()
