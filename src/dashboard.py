"""Dashboard HTML statis (tanpa server) yang dihasilkan dari artifact model."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from . import config


def _img_tag(path: Path, alt: str) -> str:
    if not Path(path).exists():
        return f'<p class="muted">[gambar tidak tersedia: {alt}]</p>'
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%".replace(".", ",")


def build_dashboard(metrics: dict, coef_df: pd.DataFrame, high_risk: pd.DataFrame) -> Path:
    m = metrics["metrics_default"]
    mr = metrics["metrics_recommended"]
    cv = metrics["cv"]

    cards = [
        ("ROC-AUC", f"{m['roc_auc']:.4f}".replace(".", ","), "kemampuan diskriminasi"),
        ("Accuracy", _pct(m["accuracy"]), f"baseline naif {_pct(metrics['baseline']['accuracy'])}"),
        ("Recall", _pct(m["recall"]), f"{_pct(mr['recall'])} pada threshold 0,30"),
        ("Precision", _pct(m["precision"]), "dari yang diprediksi churn"),
        ("F1-Score", _pct(m["f1"]), f"CV AUC {cv['mean']:.4f}".replace(".", ",")),
    ]
    card_html = "".join(
        f'<div class="card"><span>{title}</span><strong>{value}</strong><em>{note}</em></div>'
        for title, value, note in cards
    )

    top_pos = coef_df.sort_values("odds_ratio", ascending=False).head(5)
    top_neg = coef_df.sort_values("odds_ratio").head(5)

    def rows(df: pd.DataFrame) -> str:
        out = []
        for _, r in df.iterrows():
            out.append(
                f"<tr><td>{html.escape(r['label'])}</td>"
                f"<td class='num'>{r['odds_ratio']:.3f}</td>"
                f"<td class='num'>{r['coefficient']:+.3f}</td></tr>"
            )
        return "".join(out)

    risk_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(r.get('customerID', '-')))}</td>"
        f"<td>{html.escape(str(r.get('Contract', '-')))}</td>"
        f"<td class='num'>{int(r.get('tenure', 0))}</td>"
        f"<td class='num'>{r.get('MonthlyCharges', 0):.2f}</td>"
        f"<td class='num'><b>{r.get('churn_probability', 0) * 100:.1f}%</b></td>"
        f"<td><span class='badge'>{html.escape(str(r.get('risk_band', '-')))}</span></td>"
        "</tr>"
        for _, r in high_risk.head(25).iterrows()
    )

    cm = metrics["confusion"]
    html_doc = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard Churn - {config.GROUP_NAME}</title>
<style>
  :root {{ --navy:#1F3864; --blue:#2E75B6; --red:#C00000; --bg:#F4F6FA; --line:#E1E6EF; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: "Segoe UI", system-ui, sans-serif; background:var(--bg); color:#1F2430; }}
  header {{ background:linear-gradient(120deg,#1F3864,#2E75B6); color:#fff; padding:34px 40px 30px; }}
  header h1 {{ margin:0 0 6px; font-size:26px; letter-spacing:.2px; }}
  header p {{ margin:0; opacity:.85; font-size:14px; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:26px 24px 60px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px;
           box-shadow:0 1px 3px rgba(31,56,100,.06); }}
  .card span {{ display:block; font-size:11px; letter-spacing:.09em; text-transform:uppercase; color:#6B7280; }}
  .card strong {{ display:block; font-size:26px; color:var(--navy); margin:6px 0 2px; }}
  .card em {{ font-style:normal; font-size:11.5px; color:#8A93A5; }}
  section {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:20px 22px;
             margin-top:20px; box-shadow:0 1px 3px rgba(31,56,100,.05); }}
  section h2 {{ margin:0 0 14px; font-size:16px; color:var(--navy); }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  .grid2 img, section > img {{ width:100%; height:auto; border-radius:6px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; background:#EEF2F9; color:var(--navy); font-size:11px;
        letter-spacing:.06em; text-transform:uppercase; padding:9px 10px; }}
  td {{ padding:9px 10px; border-bottom:1px solid var(--line); }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr:last-child td {{ border-bottom:none; }}
  .badge {{ background:#FDECEA; color:var(--red); border-radius:999px; padding:2px 9px; font-size:11px; }}
  .muted {{ color:#8A93A5; font-size:12.5px; }}
  .kpi {{ display:flex; gap:26px; flex-wrap:wrap; font-size:13px; color:#3D4556; }}
  .kpi b {{ color:var(--navy); }}
  @media (max-width:820px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Dashboard Prediksi Customer Churn</h1>
  <p>{html.escape(config.REPORT_TITLE)} &middot; {config.GROUP_NAME} &middot; {config.YEAR}</p>
</header>
<div class="wrap">
  <div class="cards">{card_html}</div>

  <section>
    <h2>Ringkasan model</h2>
    <div class="kpi">
      <div>Data uji: <b>{metrics['split']['test_size']:,}</b> pelanggan</div>
      <div>Fitur setelah encoding: <b>{metrics['model']['n_features']}</b></div>
      <div>Confusion matrix: <b>TN {cm[0][0]} / FP {cm[0][1]} / FN {cm[1][0]} / TP {cm[1][1]}</b></div>
      <div>Threshold operasional: <b>0,30</b> (recall {_pct(mr['recall'])})</div>
    </div>
  </section>

  <section>
    <h2>Kinerja model</h2>
    <div class="grid2">
      {_img_tag(config.FIGURE_DIR / '07_roc_curve.png', 'Kurva ROC')}
      {_img_tag(config.FIGURE_DIR / '08_threshold_tuning.png', 'Threshold tuning')}
    </div>
  </section>

  <section>
    <h2>Confusion matrix & validasi silang</h2>
    <div class="grid2">
      {_img_tag(config.FIGURE_DIR / '06_confusion_matrix.png', 'Confusion matrix')}
      {_img_tag(config.FIGURE_DIR / '10_cross_validation.png', 'Cross validation')}
    </div>
  </section>

  <section>
    <h2>Faktor penentu churn (odds ratio)</h2>
    <div class="grid2">
      <table>
        <thead><tr><th>Menaikkan risiko</th><th class="num">OR</th><th class="num">Koef.</th></tr></thead>
        <tbody>{rows(top_pos)}</tbody>
      </table>
      <table>
        <thead><tr><th>Menurunkan risiko</th><th class="num">OR</th><th class="num">Koef.</th></tr></thead>
        <tbody>{rows(top_neg)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>25 pelanggan berisiko tertinggi (data uji)</h2>
    <table>
      <thead><tr><th>Customer ID</th><th>Kontrak</th><th class="num">Tenure</th>
      <th class="num">Tagihan/bln</th><th class="num">Prob. churn</th><th>Risiko</th></tr></thead>
      <tbody>{risk_rows}</tbody>
    </table>
    <p class="muted">Dihasilkan otomatis oleh <code>run_pipeline.py</code>.
    Ambang risiko: &lt; 30% rendah, 30-60% sedang, &gt; 60% tinggi.</p>
  </section>
</div>
</body>
</html>
"""
    config.DASHBOARD_FILE.write_text(html_doc, encoding="utf-8")
    return config.DASHBOARD_FILE
