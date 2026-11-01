"""Fill report/report_template.md with the executed notebook results and render it to PDF.

Every number in the report comes from artifacts/metrics.json, figures/results_table.csv or
Data_clean/manifest.csv, so nothing is typed by hand. The script also checks that the video links
open without a login, and that the PDF is 4-5 pages.

Run from the repository root after executing the notebook:  python scripts/build_report.py
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import markdown
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF, TextStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
TEMPLATE = ROOT / "report" / "report_template.md"
MD = ROOT / "report" / "Formative2_HMM_Report.md"
PDF = ROOT / "report" / "Formative2_HMM_Report.pdf"
FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
STATES = ["standing", "walking", "jumping", "still"]

# Direct Google Drive link for each video
VIDEO_LINKS = {
    "standing": "https://drive.google.com/file/d/1Tq2zAthdj9Cif8v5GVrvESlW4DmJisJH/view",
    "walking": "https://drive.google.com/file/d/1KrsBVEBJWtxzm3FCOFKxx6KnelZwWZ2k/view",
    "jumping": "https://drive.google.com/file/d/1xfj1qU6cLenPN_gOmMwR0pfqNTGwYzM5/view",
    "still": "https://drive.google.com/file/d/1uLwOWfO70HMY1DmjN6t-fPd8Pg7eEMzD/view",
}


def link_is_public(url):
    """True if the URL opens without a Google sign-in redirect, None if there is no network."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        return urllib.request.build_opener(NoRedirect).open(req, timeout=15).status == 200
    except urllib.error.HTTPError:
        return False
    except OSError:
        return None


def combine_figures():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, name in zip(axes, ["confusion_matrix.png", "transition_matrix.png", "baum_welch_convergence.png"]):
        ax.imshow(mpimg.imread(FIG / name))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "report_cm_transition.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def md_table(df):
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))


def render():
    """Fill every {{placeholder}} in the template from the saved results."""
    m = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    man = pd.read_csv(ROOT / "Data_clean" / "manifest.csv")
    impact = pd.read_csv(ROOT / "Data_clean" / "trim_rule_impact.csv")
    res = pd.read_csv(FIG / "results_table.csv")

    g = {s: man[man.activity == s] for s in STATES}
    ds = pd.DataFrame({
        "Activity": [s.capitalize() for s in STATES],
        "Files (train + test)": [f"{(g[s].split == 'train').sum()} + {(g[s].split == 'test').sum()}" for s in STATES],
        "Original (s)": [f"{g[s].original_duration_s.sum():.1f}" for s in STATES],
        "Clean train (s)": [f"{g[s][g[s].split == 'train'].duration_s.sum():.1f}" for s in STATES],
        "Clean test (s)": [f"{g[s][g[s].split == 'test'].duration_s.sum():.1f}" for s in STATES],
        "Clean total (s)": [f"{g[s].duration_s.sum():.1f}" for s in STATES],
    })
    rt = res.copy()
    rt["State (Activity)"] = rt["State (Activity)"].str.capitalize()
    for c in ["Sensitivity", "Specificity", "Overall Accuracy"]:
        rt[c] = rt[c].map("{:.3f}".format)
    pc = {r["State (Activity)"]: r for r in m["per_class"]}

    values = {k: v for k, v in m.items() if not isinstance(v, (list, dict))}
    values.update({
        "DATASET_TABLE": md_table(ds), "RESULTS_TABLE": md_table(rt),
        "n_files": len(man), "n_recordings": int(impact.recordings.sum()),
        "fixed_rule_kept": int(impact.fixed_rule_files_kept.sum()),
        "A_walk": f"{m['A_diag']['walking']:.6f}",
        "sens_still": f"{pc['still']['Sensitivity']:.3f}", "spec_standing": f"{pc['standing']['Specificity']:.3f}",
        "train_acc_init_pct": f"{100 * m['train_acc_init']:.1f}", "train_acc_bw_pct": f"{100 * m['train_acc_bw']:.1f}",
        "ll_initial": f"{m['ll_initial']:.1f}".replace("-", "−"), "ll_final": f"{m['ll_final']:.1f}".replace("-", "−"),
    })
    values.update({f"VIDEO_{s}": url for s, url in VIDEO_LINKS.items()})
    text = TEMPLATE.read_text(encoding="utf-8")
    text = re.sub(r"\{\{(\w+)\}\}", lambda mt: str(values[mt.group(1)]), text)
    assert "{{" not in text, "Unfilled placeholder in report"
    MD.write_text(text, encoding="utf-8")
    print(f"Wrote {MD.relative_to(ROOT)} from the template")
    return text


IMG_WIDTH = {  # mm
    "sample_signals.png": 145, "report_cm_transition.png": 172,
    "emission_parameters.png": 165, "decoded_sequence.png": 172}


def to_html(text):
    html = markdown.markdown(text, extensions=["tables"])

    def img(mt):
        name = Path(mt.group(1)).name
        px = round(IMG_WIDTH.get(name, 150) * 72 / 25.4)    # fpdf2 reads <img width> in px at 72 dpi
        return f'<img src="{(FIG / name).as_posix()}" width="{px}">'
    html = re.sub(r'<img alt="[^"]*" src="([^"]+)" ?/?>', img, html)
    html = html.replace("<table>", '<table width="100%" border="1">')
    html = re.sub(r"<code>(.*?)</code>", r"<font face='consolas'>\1</font>", html)
    html = re.sub(r"(?<![\"'>])(https://[^\s<]+)", r'<a href="\1">\1</a>', html)
    return html


def main():
    combine_figures()
    text = render()
    for s, url in VIDEO_LINKS.items():
        ok = link_is_public(url)
        status = "opens without login" if ok else ("UNVERIFIED (no network)" if ok is None else "NOT PUBLIC: asks for sign-in")
        print(f"Video link {s}: {status}")

    pdf = FPDF(format="A4")
    pdf.set_margins(18, 15, 18)
    pdf.set_auto_page_break(True, margin=15)
    for style, f in (("", "arial"), ("B", "arialbd"), ("I", "ariali"), ("BI", "arialbi")):
        pdf.add_font("arial", style, FONTS / f"{f}.ttf")
    for style, f in (("", "consola"), ("B", "consolab"), ("I", "consolai"), ("BI", "consolaz")):
        pdf.add_font("consolas", style, FONTS / f"{f}.ttf")
    pdf.set_font("arial", size=10)
    pdf.add_page()
    pdf.write_html(to_html(text), font_family="arial", li_prefix_color=(0, 0, 0),
                   tag_styles={"h1": TextStyle(font_family="arial", font_style="B", font_size_pt=16, t_margin=0, b_margin=2),
                               "h2": TextStyle(font_family="arial", font_style="B", font_size_pt=12.5, t_margin=4, b_margin=1.5)})
    pdf.output(PDF)
    pages = len(PdfReader(PDF).pages)
    print(f"Wrote {PDF.relative_to(ROOT)}: {pages} pages")
    assert 4 <= pages <= 5, "Report must be 4-5 pages"


if __name__ == "__main__":
    main()
