# -*- coding: utf-8 -*-
"""FondationLab : pression moyenne sous une semelle, modèle pédagogique.

Toutes les ressources sont intégrées. Aucun fichier utilisateur n'est écrit.
Les fonctions de calcul et d'export sont indépendantes de l'interface.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from html import escape
from io import BytesIO, StringIO
import math
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Flowable, PageBreak,
)


VERSION = "1.0.0"
INK = "#22272B"
LIME = "#D3F36B"
CYAN = "#007E91"
PINK = "#B82A60"
EXAMPLE = {
    "nom": "Semelle A · exemple fictif", "longueur_m": 2.0,
    "largeur_m": 1.5, "charge_kN": 300.0, "seuil_kPa": 110.0,
    "origine": "Exemple fictif",
}
FIELDS = ["nom", "longueur_m", "largeur_m", "charge_kN", "seuil_kPa"]
LIMITS = {
    "longueur_m": (0.10, 50.0), "largeur_m": (0.10, 50.0),
    "charge_kN": (0.0, 10_000_000.0), "seuil_kPa": (0.1, 100_000.0),
}
ASSUMPTIONS = (
    "Charge totale verticale centrée, incluant les poids que l'utilisateur souhaite considérer. "
    "Le poids propre de la semelle n'est pas ajouté automatiquement. "
    "Contact intégral et pression uniforme idéalisée : q = N / (L × B). "
    "Le seuil est une valeur de comparaison fournie, pas une propriété du sol calculée par l'application. "
    "Aucune vérification de portance réelle, tassement, excentricité, glissement, "
    "poinçonnement, flexion, ferraillage ou conformité réglementaire."
)


def fmt(value, digits=2):
    return f"{float(value):,.{digits}f}".replace(",", " ").replace(".", ",")


def validate_case(raw):
    if not isinstance(raw, dict):
        raise ValueError("Un cas doit être un ensemble de champs nommés.")
    name = raw.get("nom")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise ValueError("Le nom doit contenir entre 1 et 80 caractères.")
    if any(ord(c) < 32 for c in name):
        raise ValueError("Le nom contient un caractère de contrôle.")
    clean = {"nom": name.strip()}
    for key, (low, high) in LIMITS.items():
        value = raw.get(key)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} : une valeur numérique est attendue.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} : valeur manquante ou non numérique.") from None
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{key} doit être compris entre {low:g} et {high:g}.")
        clean[key] = number
    clean["origine"] = str(raw.get("origine", "Hypothèses saisies"))[:80]
    return clean


def results(case):
    c = validate_case(case)
    area = c["longueur_m"] * c["largeur_m"]
    pressure = c["charge_kN"] / area
    return {
        "surface_m2": area, "pression_kPa": pressure,
        "ratio_pct": 100 * pressure / c["seuil_kPa"],
        "ecart_kPa": c["seuil_kPa"] - pressure,
        "largeur_egalite_m": c["charge_kN"] / (c["longueur_m"] * c["seuil_kPa"]),
    }


def fingerprint(case):
    c = validate_case(case)
    return tuple(c[k] for k in FIELDS + ["origine"])


def distribution(low, mode, high):
    values = [low, mode, high]
    if any(isinstance(v, bool) for v in values):
        raise ValueError("Les charges doivent être numériques.")
    try:
        low, mode, high = map(float, values)
    except (TypeError, ValueError):
        raise ValueError("Les trois charges doivent être numériques.") from None
    if not all(math.isfinite(v) for v in (low, mode, high)):
        raise ValueError("Les charges doivent être finies.")
    if not 0 <= low <= mode <= high <= 10_000_000:
        raise ValueError("Respecter 0 ≤ minimum ≤ valeur probable ≤ maximum ≤ 10 000 000 kN.")
    if low == high:
        return None
    return stats.triang(c=(mode-low)/(high-low), loc=low, scale=high-low)


def exact_risk(case, low, mode, high):
    c = validate_case(case)
    law = distribution(low, mode, high)
    area = results(c)["surface_m2"]
    threshold = c["seuil_kPa"] * area
    if law is None:
        return {"probabilite": float(low > threshold), "moyenne": low/area,
                "ecart_type": 0.0, "q05": low/area, "q50": low/area, "q95": low/area}
    return {"probabilite": float(law.sf(threshold)), "moyenne": float(law.mean()/area),
            "ecart_type": float(law.std()/area),
            **{f"q{p:02}": float(law.ppf(p/100)/area) for p in (5, 50, 95)}}


def simulate(case, low, mode, high, count=20000, seed=42):
    c = validate_case(case)
    distribution(low, mode, high)
    if isinstance(count, bool) or int(count) != count or not 1000 <= count <= 100000:
        raise ValueError("Choisir entre 1 000 et 100 000 simulations entières.")
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed <= 2147483647:
        raise ValueError("La graine doit être un entier entre 0 et 2 147 483 647.")
    rng = np.random.default_rng(int(seed))
    loads = np.full(int(count), float(low)) if low == high else rng.triangular(low, mode, high, int(count))
    pressures = loads / results(c)["surface_m2"]
    hits = int(np.count_nonzero(pressures > c["seuil_kPa"]))
    interval = stats.binomtest(hits, len(pressures)).proportion_ci(0.95, method="exact")
    return {
        "empreinte": fingerprint(c), "minimum": float(low), "probable": float(mode),
        "maximum": float(high), "nombre": int(count), "graine": int(seed),
        "charges": loads, "pressions": pressures, "depassements": hits,
        "probabilite": hits/len(pressures), "ic_bas": float(interval.low),
        "ic_haut": float(interval.high), "exact": exact_risk(c, low, mode, high),
    }


cached_simulate = st.cache_data(show_spinner=False, max_entries=12)(simulate)


def case_frame(cases):
    records = []
    for case in cases:
        c = validate_case(case)
        records.append({**c, **results(c)})
    return pd.DataFrame(records)


def safe_sheet(frame):
    copy = frame.copy()
    for col in copy.select_dtypes(include=["object", "str"]).columns:
        copy[col] = copy[col].map(
            lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
        )
    return copy


def csv_bytes(frame):
    return safe_sheet(frame).to_csv(index=False).encode("utf-8-sig")


def excel_bytes(frame):
    stream = BytesIO()
    with pd.ExcelWriter(stream, engine="openpyxl") as writer:
        safe_sheet(frame).to_excel(writer, index=False, sheet_name="Semelles")
        sheet = writer.sheets["Semelles"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for col in sheet.columns:
            sheet.column_dimensions[col[0].column_letter].width = min(42, max(18, len(str(col[0].value))+3))
    return stream.getvalue()


def parse_import(data, filename):
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("Le fichier dépasse 5 Mo.")
    suffix = filename.rsplit(".", 1)[-1].lower()
    try:
        if suffix == "csv":
            text = data.decode("utf-8-sig")
            frame = pd.read_csv(StringIO(text), sep=None, engine="python", nrows=101, dtype=str)
        elif suffix == "xlsx":
            with zipfile.ZipFile(BytesIO(data)) as archive:
                if sum(i.file_size for i in archive.infolist()) > 20 * 1024 * 1024:
                    raise ValueError("Le classeur décompressé dépasse 20 Mo.")
            frame = pd.read_excel(BytesIO(data), engine="openpyxl", nrows=101, dtype=str)
        else:
            raise ValueError("Utiliser un fichier CSV UTF-8 ou Excel .xlsx.")
    except ValueError:
        raise
    except Exception:
        raise ValueError("Fichier illisible : vérifier le format, l'encodage UTF-8 et la première feuille Excel.") from None
    frame.columns = [str(c).strip() for c in frame.columns]
    if not set(FIELDS).issubset(frame.columns):
        raise ValueError("Colonnes requises : " + ", ".join(FIELDS))
    if frame.empty or len(frame) > 100:
        raise ValueError("Importer entre 1 et 100 lignes.")
    cases, seen = [], set()
    for i, row in frame.iterrows():
        raw = row.to_dict()
        for k in LIMITS:
            if isinstance(raw.get(k), str):
                raw[k] = raw[k].strip().replace(",", ".")
        raw["origine"] = "Import utilisateur"
        try:
            case = validate_case(raw)
        except ValueError as exc:
            raise ValueError(f"Ligne {i+2} : {exc}") from None
        if case["nom"].casefold() in seen:
            raise ValueError(f"Ligne {i+2} : nom dupliqué.")
        seen.add(case["nom"].casefold())
        cases.append(case)
    return cases


CSS = """
<style>
:root { --fl-ink:#22272b; --fl-lime:#d3f36b; --fl-cyan:#007e91; }
.stApp { background:#f2f4f3; color:#22272b; }
[data-testid="stHeader"] { background:#f2f4f3; }
[data-testid="stMainBlockContainer"] { max-width:1550px; padding:4.8rem 2.4rem 3rem; }
[data-testid="stSidebar"] { background:#22272b; border-right:0; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#f3f6f2; }
[data-testid="stSidebar"] button { background:transparent; color:#e7ede7; border:1px solid #525a5d; }
[data-testid="stSidebar"] button:hover { background:#363f3d; border-color:#d3f36b; color:#fff; }
[data-testid="stSidebar"] button[kind="primary"] { background:#d3f36b; color:#22272b; border-color:#d3f36b; }
.fl-side-brand { font-size:24px; font-weight:800; color:#d3f36b; padding:10px 0 5px; }
.fl-side-sub { color:#c1cbc5; font-size:12px; margin-bottom:28px; }
.fl-top { display:flex; justify-content:space-between; align-items:center; gap:16px;
  padding:22px 26px; background:#d3f36b; border-bottom:4px solid #22272b; margin-bottom:20px; }
.fl-top h1 { font-family:Arial,sans-serif; font-size:36px; line-height:1.12; font-weight:800;
  margin:0; letter-spacing:0; color:#22272b; overflow-wrap:anywhere; }
.fl-top p { margin:6px 0 0; font-size:14px; color:#364425; }
.fl-tag { padding:7px 0; border-bottom:2px solid #22272b; font:600 12px Arial; white-space:normal; }
.fl-context { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px;
  margin:0 0 15px; font-size:13px; color:#505d55; }
.fl-context b { color:#22272b; }
.fl-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:6px 0 24px; }
.fl-stat { padding:16px 18px; border-radius:6px; border-top:4px solid #007e91; background:#fff; min-width:0; }
.fl-stat:nth-child(2) { background:#22272b; color:#fff; border-color:#d3f36b; }
.fl-stat:nth-child(3) { background:#e1f0f2; border-color:#007e91; }
.fl-stat:nth-child(4) { background:#f6e8ee; border-color:#b82a60; }
.fl-stat span { display:block; font-size:12px; margin-bottom:7px; }
.fl-stat strong { display:block; font-size:27px; line-height:1.25; overflow-wrap:anywhere; }
.fl-stat small { display:block; font-size:12px; margin-top:6px; opacity:.85; }
.fl-note { border-left:4px solid #007e91; padding:12px 16px; background:#e1f0f2; margin:10px 0; font-size:14px; }
.fl-note.rose { background:#f6e8ee; border-color:#b82a60; }
.fl-equation { padding:18px 22px; background:#e5ecd9; border-left:4px solid #607d24;
  font:600 17px/1.65 Consolas,monospace; overflow-wrap:anywhere; margin:14px 0; }
.fl-footer { padding:20px 0 0; border-top:1px solid #ccd4cd; color:#59665f; font-size:12px; margin-top:26px; }
h2 { font-size:24px!important; letter-spacing:0!important; }
h3 { font-size:19px!important; letter-spacing:0!important; }
[data-testid="stWidgetLabel"] p { color:#27362f; font-size:14px; }
[data-testid="stNumberInputContainer"], [data-baseweb="input"], [data-baseweb="select"] > div { background:#fff; }
.stButton button, .stDownloadButton button { border-radius:5px; min-height:42px; }
.stButton button p, .stDownloadButton button p { white-space:normal; overflow-wrap:anywhere; }
.stButton button:disabled { opacity:.5!important; cursor:not-allowed!important; }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
  background:#007e91; color:#fff; border-color:#007e91; }
.stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover { background:#006172; }
[data-testid="stForm"] { background:#fff; border-radius:6px; border:1px solid #cbd4ce; }
[data-testid="stAlert"] { border-radius:5px; }
[data-baseweb="tab-highlight"] { background:#007e91; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p { color:#fff; }
.stApp [data-testid="stSidebar"] button[kind="primary"] { background:#d3f36b; color:#22272b; border-color:#d3f36b; }
.stApp [data-testid="stSidebar"] button[kind="primary"] p,
.stApp [data-testid="stSidebar"] button[kind="primary"] span { color:#22272b!important; }
[data-testid="stFormSubmitButton"] button { background:#007e91; border-color:#007e91; color:#fff; }
[data-testid="stFormSubmitButton"] button:hover { background:#006172; border-color:#006172; color:#fff; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background:#e1f0f2!important; border-color:#007e91!important; color:#005b69!important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] p { color:#005b69; }
@media(max-width:1100px) {
  [data-testid="stMainBlockContainer"] { padding:4.5rem 1.2rem 2rem; }
  .fl-stat { padding:13px 12px; } .fl-stat strong { font-size:23px; }
}
@media(max-width:720px) {
  .fl-top { padding:18px; align-items:flex-start; flex-direction:column; gap:6px; }
  .fl-top h1 { font-size:30px; } .fl-top p { font-size:13px; }
  .fl-stats { grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
  .fl-stat strong { font-size:23px; }
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:100%!important; width:100%!important; }
}
@media(prefers-reduced-motion:reduce) { * { animation:none!important; transition:none!important; } }
</style>
"""


def diagram_svg(case, view="Coupe"):
    c = validate_case(case)
    r = results(c)
    label = escape(f"N = {fmt(c['charge_kN'])} kN")
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 410" role="img"
      aria-label="Schéma de la semelle, {escape(view)}, charge verticale centrée">
      <defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#46514d" stroke-width="0.5"/>
      </pattern><pattern id="soil" width="18" height="18" patternUnits="userSpaceOnUse">
      <path d="M0 18 L18 0" stroke="#5b7164" stroke-width="1"/></pattern>
      <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0 0 L6 3 L0 6" fill="#d3f36b"/></marker></defs>
      <rect width="720" height="410" fill="#22272b"/>
      <rect x="0" y="0" width="720" height="410" fill="url(#grid)"/>
      <g font-family="Arial,sans-serif" fill="#eef5ec">
      <text x="24" y="30" font-size="12" fill="#d3f36b">{escape(view.upper())} / CHARGE CENTRÉE</text>''']
    if view == "Plan":
        factor = min(430/c["longueur_m"], 230/c["largeur_m"])
        w, h = c["longueur_m"]*factor, c["largeur_m"]*factor
        x, y = 360-w/2, 210-h/2
        parts.append(f'''<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#d3f36b" fill-opacity="0.15" stroke="#d3f36b" stroke-width="3"/>
          <path d="M{x} {y-12} H{x+w}" stroke="#78cfda"/>
          <text x="360" y="{y-22}" text-anchor="middle" font-size="17">L = {fmt(c['longueur_m'])} m</text>
          <text x="360" y="365" text-anchor="middle" font-size="17">B = {fmt(c['largeur_m'])} m · A = {fmt(r['surface_m2'])} m²</text>
          <circle cx="360" cy="210" r="15" fill="#78cfda"/>
          <path d="M352 202 L368 218 M368 202 L352 218" stroke="#22272b" stroke-width="3"/>
          <text x="360" y="400" text-anchor="middle" font-size="13" fill="#c4d0c8">Force perpendiculaire au plan : {label}</text>''')
    else:
        parts.append(f'''<rect x="65" y="247" width="590" height="113" fill="#34473c"/>
          <rect x="65" y="247" width="590" height="113" fill="url(#soil)"/>
          <path d="M65 247 H655" stroke="#78cfda" stroke-width="2"/>
          <rect x="160" y="200" width="400" height="47" fill="#d3f36b" stroke="#efffc3" stroke-width="2"/>
          <rect x="315" y="115" width="90" height="85" fill="#b3c2bb"/>
          <path d="M360 57 V151" stroke="#d3f36b" stroke-width="5" marker-end="url(#arrow)"/>
          <text x="389" y="86" font-size="19">{label}</text>
          <text x="360" y="230" fill="#22272b" text-anchor="middle" font-size="17">B = {fmt(c['largeur_m'])} m</text>
          <text x="360" y="338" text-anchor="middle" fill="#e4f0e6" font-size="20">q = {fmt(r['pression_kPa'])} kPa</text>
          <text x="360" y="396" text-anchor="middle" font-size="12" fill="#c4d0c8">Coupe schématique · épaisseurs non calculées · L = {fmt(c['longueur_m'])} m</text>''')
        for x in range(190, 550, 45):
            parts.append(f'<path d="M{x} 300 V258" stroke="#d3f36b" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append("</g></svg>")
    return "".join(parts)


def compact_diagram(case, view):
    c, r = validate_case(case), results(case)
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 330" role="img"
      aria-label="Schéma compact de la semelle, {escape(view)}">
      <defs><pattern id="smallsoil" width="14" height="14" patternUnits="userSpaceOnUse">
      <path d="M0 14 L14 0" stroke="#5b7164"/></pattern></defs>
      <rect width="360" height="330" fill="#22272b"/>
      <g font-family="Arial,sans-serif" fill="#eef5ec">
      <text x="16" y="27" font-size="14" fill="#d3f36b">{escape(view.upper())} / CHARGE CENTRÉE</text>''']
    if view == "Plan":
        factor = min(260/c["longueur_m"], 145/c["largeur_m"])
        w, h = c["longueur_m"]*factor, c["largeur_m"]*factor
        x, y = 180-w/2, 152-h/2
        parts.append(f'''<text x="180" y="57" text-anchor="middle" font-size="17">L = {fmt(c['longueur_m'])} m</text>
          <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#d3f36b" fill-opacity="0.15" stroke="#d3f36b" stroke-width="2"/>
          <circle cx="180" cy="152" r="12" fill="#78cfda"/>
          <path d="M173 145 L187 159 M187 145 L173 159" stroke="#22272b" stroke-width="2"/>
          <text x="180" y="254" text-anchor="middle" font-size="17">B = {fmt(c['largeur_m'])} m</text>
          <text x="180" y="282" text-anchor="middle" font-size="17">A = {fmt(r['surface_m2'])} m²</text>
          <text x="180" y="312" text-anchor="middle" font-size="15">N = {fmt(c['charge_kN'])} kN</text>''')
    else:
        parts.append(f'''<rect x="22" y="194" width="316" height="76" fill="#34473c"/>
          <rect x="22" y="194" width="316" height="76" fill="url(#smallsoil)"/>
          <rect x="65" y="158" width="230" height="36" fill="#d3f36b"/>
          <rect x="157" y="100" width="46" height="58" fill="#b3c2bb"/>
          <path d="M180 70 V124 M172 114 L180 124 L188 114" fill="none" stroke="#d3f36b" stroke-width="3"/>
          <text x="180" y="57" text-anchor="middle" font-size="17">N = {fmt(c['charge_kN'])} kN</text>
          <text x="180" y="181" text-anchor="middle" fill="#22272b" font-size="17">B = {fmt(c['largeur_m'])} m</text>
          <text x="180" y="247" text-anchor="middle" font-size="19">q = {fmt(r['pression_kPa'])} kPa</text>
          <text x="180" y="294" text-anchor="middle" font-size="14">L = {fmt(c['longueur_m'])} m · coupe schématique</text>
          <text x="180" y="316" text-anchor="middle" font-size="13" fill="#c4d0c8">Épaisseurs non calculées</text>''')
        for x in range(90, 290, 40):
            parts.append(f'<path d="M{x} 224 V204 M{x-5} 211 L{x} 204 L{x+5} 211" fill="none" stroke="#d3f36b" stroke-width="2"/>')
    parts.append("</g></svg>")
    return "".join(parts)


_DRAWING = st.components.v2.component(
    "fondationlab_drawing",
    html='<div id="fl-drawing"></div>',
    css="""#fl-drawing{width:100%;line-height:0;container-type:inline-size}
    #fl-drawing svg{width:100%;height:auto;display:block}
    #fl-drawing .fl-mobile{display:none}
    @container(max-width:500px){#fl-drawing .fl-wide{display:none}#fl-drawing .fl-mobile{display:block}}""",
    js="export default function({data,parentElement}){parentElement.querySelector('#fl-drawing').innerHTML=data.svg;}",
)


def drawing(case, view, key):
    content = (f'<div class="fl-wide">{diagram_svg(case, view)}</div>'
               f'<div class="fl-mobile">{compact_diagram(case, view)}</div>')
    _DRAWING(data={"svg": content}, key=key, height="content")


def chart(fig, key, alt, height=310):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=25, r=15, t=20, b=45), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial", size=13, color=INK),
        legend=dict(orientation="h", y=-0.25, x=0),
    )
    fig.update_xaxes(automargin=True, gridcolor="#d8dfda")
    fig.update_yaxes(automargin=True, gridcolor="#d8dfda")
    st.plotly_chart(fig, width="stretch", theme=None, key=key, alt=alt,
                    config={"displaylogo": False, "scrollZoom": False, "responsive": True})


def stats_strip(items):
    tiles = "".join(
        f'<div class="fl-stat"><span>{escape(label)}</span><strong>{escape(value)}</strong>'
        f'<small>{escape(unit)}</small></div>' for label, value, unit in items
    )
    st.html(f'<div class="fl-stats">{tiles}</div>')


def note(text, rose=False):
    st.html(f'<div class="fl-note {"rose" if rose else ""}">{escape(text)}</div>')


def summary(case):
    r = results(case)
    stats_strip([
        ("Charge totale N", fmt(case["charge_kN"]), "kN · verticale centrée"),
        ("Pression moyenne q", fmt(r["pression_kPa"]), "kPa = kN/m²"),
        ("Surface de contact A", fmt(r["surface_m2"]), "m² · longueur × largeur"),
        ("Pression / seuil", fmt(r["ratio_pct"], 1) + " %", "ratio de comparaison, pas de sécurité"),
    ])


class FootingDrawing(Flowable):
    def __init__(self, case):
        super().__init__()
        self.case = case
        self.width, self.height = 455, 170

    def draw(self):
        c, r = self.canv, results(self.case)
        c.setFillColor(colors.HexColor(INK))
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#53675A"))
        c.rect(25, 18, 405, 55, fill=1, stroke=0)
        c.setFillColor(colors.HexColor(LIME))
        c.rect(80, 73, 295, 25, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#AEBDB5"))
        c.rect(205, 98, 45, 40, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor(LIME))
        c.setLineWidth(3)
        c.line(227, 157, 227, 118)
        c.line(227, 118, 221, 126)
        c.line(227, 118, 233, 126)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 10)
        c.drawString(257, 143, f"N = {fmt(self.case['charge_kN'])} kN")
        c.drawCentredString(227, 43, f"q = {fmt(r['pression_kPa'])} kPa")
        c.setFillColor(colors.HexColor(INK))
        c.drawCentredString(227, 81, f"B = {fmt(self.case['largeur_m'])} m")


def pdf_report(case, comparisons=None, risk=None):
    c = validate_case(case)
    if risk and tuple(risk["empreinte"]) != fingerprint(c):
        risk = None
    stream = BytesIO()
    doc = SimpleDocTemplate(stream, pagesize=(21*cm, 29.7*cm),
                            leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.8*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="FLTitle", fontName="Helvetica-Bold", fontSize=25,
                              leading=30, textColor=colors.HexColor(INK), spaceAfter=8))
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 15
    styles["BodyText"].spaceAfter = 8
    styles["Heading2"].textColor = colors.HexColor(CYAN)
    styles["Heading2"].spaceBefore = 14

    def p(text, style="BodyText"):
        return Paragraph(escape(str(text)), styles[style])

    def table(rows, widths=None):
        elements = [[p(v) for v in row] for row in rows]
        t = Table(elements, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EED9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CAD4CC")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    r = results(c)
    story = [p("FondationLab", "FLTitle"), p("Rapport pédagogique · " + datetime.now().strftime("%d/%m/%Y %H:%M")),
             p(c["nom"], "Heading2"), p("Origine : " + c["origine"]),
             table([["Donnée", "Valeur"], ["Longueur L", fmt(c["longueur_m"])+" m"],
                    ["Largeur B", fmt(c["largeur_m"])+" m"], ["Charge totale N", fmt(c["charge_kN"])+" kN"],
                    ["Seuil de comparaison fourni", fmt(c["seuil_kPa"])+" kPa"]], [260, 195]),
             Spacer(1, 12), FootingDrawing(c),
             p("Calcul de la pression moyenne", "Heading2"),
             p(f"A = L × B = {fmt(c['longueur_m'])} × {fmt(c['largeur_m'])} = {fmt(r['surface_m2'])} m²."),
             p(f"q = N / A = {fmt(c['charge_kN'])} / {fmt(r['surface_m2'])} = {fmt(r['pression_kPa'])} kPa."),
             p(f"q / seuil = {fmt(r['ratio_pct'])} %. Écart seuil - q = {fmt(r['ecart_kPa'])} kPa."),
             p("Ce ratio ne constitue pas un coefficient de sécurité. La coupe est schématique ; les épaisseurs ne sont pas calculées.")]
    if comparisons:
        story.extend([PageBreak(), p("Comparaison de semelles", "Heading2"),
                      p("Chaque ligne conserve sa propre charge et son propre seuil de comparaison."),
                      table([["Cas", "L × B (m)", "N (kN)", "q (kPa)", "Seuil (kPa)"]] + [
                          [x["nom"], f"{fmt(x['longueur_m'])} × {fmt(x['largeur_m'])}",
                           fmt(x["charge_kN"]), fmt(results(x)["pression_kPa"]), fmt(x["seuil_kPa"])]
                          for x in comparisons], [130, 95, 75, 75, 80])])
    if risk:
        e = risk["exact"]
        story.extend([PageBreak(), p("Incertitude sur la charge", "Heading2"),
                      p("Résultats simulés : aucune mesure de terrain n'est créée par cette simulation."),
                      p(f"Loi triangulaire : minimum {fmt(risk['minimum'])}, valeur probable {fmt(risk['probable'])}, maximum {fmt(risk['maximum'])} kN."),
                      p(f"{risk['nombre']} tirages ; graine {risk['graine']}. Dimensions et seuil constants."),
                      table([["Indicateur", "Valeur"],
                             ["P(q > seuil), théorie", fmt(e["probabilite"]*100, 3)+" %"],
                             ["P(q > seuil), simulation", fmt(risk["probabilite"]*100, 3)+" %"],
                             ["IC binomial exact à 95 %", f"[{fmt(100*risk['ic_bas'], 3)} ; {fmt(100*risk['ic_haut'], 3)}] %"],
                             ["Pression moyenne théorique", fmt(e["moyenne"])+" kPa"],
                             ["Écart-type théorique", fmt(e["ecart_type"])+" kPa"],
                             ["Quantile théorique 95 %", fmt(e["q95"])+" kPa"]], [260, 195]),
                      p("L'intervalle quantifie l'erreur des tirages Monte-Carlo, pas la fiabilité des hypothèses. Dépasser le seuil fourni n'est pas une probabilité d'effondrement.")])
    story.extend([Spacer(1, 14), p("Hypothèses et limites", "Heading2"), p(ASSUMPTIONS),
                  p("Référence technique : FHWA, Soils and Foundations, NHI-06-089. Documentation de la loi triangulaire : SciPy, scipy.stats.triang.")])

    def footer(canvas, document):
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#59665F"))
        canvas.drawString(2*cm, 1*cm, "FondationLab · Outil pédagogique · v" + VERSION)
        canvas.drawRightString(19*cm, 1*cm, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()


def active_case():
    return st.session_state.case or deepcopy(EXAMPLE)


def set_case(case):
    st.session_state.case = validate_case(case)
    st.session_state.risk = None
    st.session_state.comparison = None
    st.session_state.pop("pdf", None)
    st.session_state.revision += 1


def case_form(case):
    revision = st.session_state.revision
    with st.form("case_form_"+str(revision), border=True):
        st.subheader("Données du cas")
        name = st.text_input("Nom du cas", value=case["nom"], max_chars=80, key=f"name_{revision}")
        a, b = st.columns(2)
        length = a.number_input("Longueur L (m)", .1, 50.0, float(case["longueur_m"]), .1, format="%.2f", key=f"length_{revision}")
        width = b.number_input("Largeur B (m)", .1, 50.0, float(case["largeur_m"]), .1, format="%.2f", key=f"width_{revision}")
        load = st.number_input("Charge verticale totale N (kN)", 0.0, 10000000.0, float(case["charge_kN"]), 10.0, key=f"load_{revision}",
                               help="Inclure les poids que vous souhaitez considérer, dont celui de la semelle si connu. Aucun poids n'est ajouté automatiquement.")
        threshold = st.number_input("Seuil de comparaison fourni (kPa)", .1, 100000.0, float(case["seuil_kPa"]), 10.0, key=f"threshold_{revision}",
                                    help="Valeur d'exercice ou donnée fournie. L'application ne détermine pas la résistance du sol.")
        submitted = st.form_submit_button("Calculer la pression", icon=":material/calculate:", type="primary", width="stretch")
    if submitted:
        try:
            set_case(dict(nom=name, longueur_m=length, largeur_m=width, charge_kN=load,
                          seuil_kPa=threshold, origine="Hypothèses saisies"))
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def dashboard(case):
    summary(case)
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        st.subheader("La semelle en un regard")
        view = st.segmented_control("Vue du schéma", ["Coupe", "Plan"], default="Coupe", key="drawing_view", label_visibility="collapsed") or "Coupe"
        drawing(case, view, "dashboard_drawing")
    with right:
        case_form(case)
    r = results(case)
    if r["pression_kPa"] > case["seuil_kPa"]:
        note(f"La pression moyenne dépasse le seuil fourni de {fmt(-r['ecart_kPa'])} kPa. Cela ne suffit pas à conclure sur la sécurité réelle.", rose=True)
    else:
        note(f"La pression moyenne est inférieure ou égale au seuil fourni. Écart : {fmt(r['ecart_kPa'])} kPa. Ce résultat ne valide pas la fondation.")
    st.subheader("Que se passe-t-il si la largeur change ?")
    a, b = st.columns([2, 1], gap="large")
    with a:
        extent = max(3.0, min(50.0, case["largeur_m"] * 2))
        widths = np.linspace(.1, extent, 200)
        fig = go.Figure(go.Scatter(x=widths, y=case["charge_kN"]/(case["longueur_m"]*widths),
                                    mode="lines", name="Pression moyenne", line=dict(color=CYAN, width=3),
                                    hovertemplate="B : %{x:.2f} m<br>q : %{y:.2f} kPa<extra></extra>"))
        fig.add_hline(y=case["seuil_kPa"], line_dash="dash", line_color=PINK)
        fig.add_trace(go.Scatter(x=[case["largeur_m"]], y=[r["pression_kPa"]], mode="markers",
                                 marker=dict(color=PINK, size=11), name="Cas actif"))
        fig.update_layout(xaxis_title="Largeur B (m)", yaxis_title="Pression q (kPa)")
        fig.update_yaxes(range=[0, max(case["seuil_kPa"], r["pression_kPa"], 1)*2.2])
        chart(fig, "sensitivity", "Pression moyenne selon la largeur, longueur et charge constantes")
        st.caption("Trait pointillé : seuil fourni. Le haut de la courbe peut sortir de l'échelle ; les valeurs restent accessibles au survol.")
    with b:
        st.subheader("Calcul détaillé")
        st.html(f'<div class="fl-equation">A = {fmt(case["longueur_m"])} × {fmt(case["largeur_m"])}<br>= {fmt(r["surface_m2"])} m²<br><br>q = {fmt(case["charge_kN"])} / {fmt(r["surface_m2"])}<br>= {fmt(r["pression_kPa"])} kPa</div>')
        st.caption(f"À charge et longueur constantes, q = seuil pour B = {fmt(r['largeur_egalite_m'], 3)} m. C'est une égalité algébrique, pas une dimension recommandée.")
    with st.expander("Hypothèses du modèle"):
        st.markdown(ASSUMPTIONS)


def comparison_page(case):
    st.header("Comparer des semelles")
    st.caption("Comparaison à charge et seuil identiques au cas actif. L'import permet aussi de comparer des cas de charges différents.")
    rev = st.session_state.revision
    with st.form("compare_form_"+str(rev)):
        a, b = st.columns(2)
        length = a.number_input("Longueur de la variante (m)", .1, 50.0, float(case["longueur_m"]), .1, key=f"cmp_length_{rev}")
        width = b.number_input("Largeur de la variante (m)", .1, 50.0, 2.0, .1, key=f"cmp_width_{rev}")
        name = st.text_input("Nom de la variante", "Semelle B", max_chars=80)
        compare = st.form_submit_button("Comparer les deux semelles", type="primary", icon=":material/compare_arrows:")
    if compare:
        try:
            variant = validate_case({**case, "nom": name, "longueur_m": length, "largeur_m": width, "origine": "Variante calculée"})
            st.session_state.comparison = [deepcopy(case), variant]
            st.session_state.pop("pdf", None)
        except ValueError as exc:
            st.error(str(exc))
    cases = st.session_state.comparison
    if not cases:
        note("Aucune comparaison enregistrée.")
        return
    r0, r1 = results(cases[0]), results(cases[1])
    stats_strip([("Pression A", fmt(r0["pression_kPa"]), "kPa"), ("Pression B", fmt(r1["pression_kPa"]), "kPa"),
                 ("Écart B - A", fmt(r1["pression_kPa"]-r0["pression_kPa"]), "kPa"),
                 ("Surface B - A", fmt(r1["surface_m2"]-r0["surface_m2"]), "m²")])
    a, b = st.columns(2)
    for column, c, key in zip((a, b), cases, ("compare_a", "compare_b")):
        with column:
            st.subheader(c["nom"])
            drawing(c, "Plan", key)
    table = case_frame(cases)
    st.dataframe(table, hide_index=True, width="stretch", alt="Caractéristiques et résultats des deux semelles")
    if r0["pression_kPa"] > 0:
        note(f"Variation de pression B par rapport à A : {fmt(100*(r1['pression_kPa']/r0['pression_kPa']-1), 1)} %. Les dimensions seules ne permettent pas de juger la faisabilité.")


def risk_page(case):
    st.header("Une charge incertaine, quel dépassement ?")
    st.caption("Seule la charge varie. Les dimensions et le seuil du cas actif restent constants.")
    rev = st.session_state.revision
    with st.form("risk_form_"+str(rev)):
        a, b, c = st.columns(3)
        low = a.number_input("Charge minimale (kN)", 0.0, 10000000.0, float(case["charge_kN"]*.8), 10.0, key=f"risk_low_{rev}")
        mode = b.number_input("Charge la plus probable (kN)", 0.0, 10000000.0, float(case["charge_kN"]), 10.0, key=f"risk_mode_{rev}")
        high = c.number_input("Charge maximale (kN)", 0.0, 10000000.0, float(min(10000000, case["charge_kN"]*1.2)), 10.0, key=f"risk_high_{rev}")
        a, b = st.columns(2)
        count = a.select_slider("Nombre de tirages", [1000, 5000, 20000, 50000, 100000], value=20000)
        seed = b.number_input("Graine reproductible", 0, 2147483647, 42, 1)
        run = st.form_submit_button("Simuler les charges", type="primary", icon=":material/play_arrow:")
    if run:
        try:
            with st.spinner("Simulation des charges…"):
                st.session_state.risk = cached_simulate(case, low, mode, high, count, seed)
                st.session_state.pop("pdf", None)
        except ValueError as exc:
            st.error(str(exc))
    risk = st.session_state.risk
    if not risk or tuple(risk["empreinte"]) != fingerprint(case):
        note("Aucune simulation pour ce cas. La loi triangulaire représente une hypothèse d'exercice, pas des observations.")
        return
    exact = risk["exact"]
    stats_strip([
        ("Dépassement simulé", fmt(100*risk["probabilite"], 2)+" %", f"{risk['depassements']} / {risk['nombre']} tirages"),
        ("Dépassement théorique", fmt(100*exact["probabilite"], 2)+" %", "P(q > seuil fourni)"),
        ("Pression moyenne", fmt(exact["moyenne"]), "kPa · valeur théorique"),
        ("Quantile 95 %", fmt(exact["q95"]), "kPa · valeur théorique"),
    ])
    note(f"IC binomial exact à 95 % : [{fmt(risk['ic_bas']*100, 3)} % ; {fmt(risk['ic_haut']*100, 3)} %]. Cet intervalle concerne l'erreur Monte-Carlo, pas l'incertitude sur le modèle.")
    a, b = st.columns(2)
    with a:
        st.subheader("Répartition des pressions simulées")
        fig = go.Figure(go.Histogram(x=risk["pressions"], nbinsx=45, marker_color=CYAN, name="Tirages simulés", histnorm="probability"))
        fig.add_vline(x=case["seuil_kPa"], line_color=PINK, line_dash="dash")
        fig.update_layout(xaxis_title="Pression q (kPa)", yaxis_title="Proportion par classe")
        chart(fig, "risk_histogram", "Histogramme des pressions simulées et seuil fourni")
    with b:
        st.subheader("Probabilité de dépasser une pression")
        area = results(case)["surface_m2"]
        x = np.linspace(risk["minimum"]/area, risk["maximum"]/area, 200)
        law = distribution(risk["minimum"], risk["probable"], risk["maximum"])
        if law is None:
            x = np.array([max(0, x[0]-1), x[0], x[0], x[0]+1])
            y = np.array([1, 1, 0, 0])
        else:
            y = law.sf(x*area)
        fig = go.Figure(go.Scatter(x=x, y=y*100, mode="lines", line=dict(color=PINK, width=3), name="Loi théorique"))
        fig.add_trace(go.Scatter(x=[case["seuil_kPa"]], y=[100*exact["probabilite"]], mode="markers", marker=dict(size=10, color=CYAN), name="Seuil fourni"))
        fig.update_layout(xaxis_title="Pression testée (kPa)", yaxis_title="Probabilité de dépassement (%)")
        chart(fig, "risk_survival", "Probabilité théorique de dépassement selon la pression testée")
    simulated = risk["pressions"]
    table = pd.DataFrame({"Indicateur": ["Moyenne", "Écart-type", "Quantile 5 %", "Médiane", "Quantile 95 %"],
                          "Théorie (kPa)": [exact["moyenne"], exact["ecart_type"], exact["q05"], exact["q50"], exact["q95"]],
                          "Simulation (kPa)": [simulated.mean(), simulated.std(ddof=1), *np.quantile(simulated, [.05, .5, .95])]})
    st.dataframe(table, hide_index=True, width="stretch", alt="Statistiques théoriques et simulées")
    st.caption(f"Hypothèse enregistrée : triangle ({fmt(risk['minimum'])} ; {fmt(risk['probable'])} ; {fmt(risk['maximum'])}) kN. {risk['nombre']} tirages, graine {risk['graine']}.")
    note("Cette probabilité n'est pas une probabilité d'effondrement. Aucune résistance du sol aléatoire n'est modélisée.", rose=True)


def data_page(case):
    st.header("Données et rapports")
    tabs = st.tabs(["Exporter", "Importer"])
    with tabs[0]:
        st.subheader("Le cas actif et ses résultats")
        frame = case_frame([case])
        st.dataframe(frame, hide_index=True, width="stretch", alt="Données et résultats exportés du cas actif")
        with st.container(horizontal=True):
            st.download_button("CSV", csv_bytes(frame), "FondationLab_cas.csv", "text/csv", icon=":material/download:")
            st.download_button("Excel", excel_bytes(frame), "FondationLab_cas.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:")
        if st.button("Préparer le rapport PDF", type="primary", icon=":material/picture_as_pdf:"):
            st.session_state.pdf = {"empreinte": fingerprint(case), "data": pdf_report(case, st.session_state.comparison, st.session_state.risk)}
        report = st.session_state.get("pdf")
        if report and tuple(report["empreinte"]) == fingerprint(case):
            st.download_button("Télécharger le rapport PDF", report["data"], "FondationLab_rapport.pdf", "application/pdf", icon=":material/download:")
        comparison = st.session_state.comparison
        if comparison:
            st.subheader("Comparaison enregistrée")
            st.download_button("Exporter la comparaison CSV", csv_bytes(case_frame(comparison)), "FondationLab_comparaison.csv", "text/csv", icon=":material/download:")
        risk = st.session_state.risk
        if risk:
            st.subheader("Tirages simulés")
            frame = pd.DataFrame({"tirage": np.arange(1, risk["nombre"]+1), "charge_simulee_kN": risk["charges"],
                                  "pression_simulee_kPa": risk["pressions"], "depasse_seuil": risk["pressions"] > case["seuil_kPa"]})
            st.download_button("Exporter les tirages CSV", csv_bytes(frame), "FondationLab_simulations.csv", "text/csv", icon=":material/download:")
    with tabs[1]:
        st.subheader("Importer des hypothèses")
        st.caption("CSV UTF-8 ou Excel .xlsx : première feuille, 100 lignes et 5 Mo maximum. L'import est prévisualisé avant toute utilisation.")
        template = pd.DataFrame([
            {k: EXAMPLE[k] for k in FIELDS},
            {**{k: EXAMPLE[k] for k in FIELDS}, "nom": "Semelle B · exemple fictif", "largeur_m": 2.0},
        ])
        st.download_button("Modèle CSV fictif", csv_bytes(template), "FondationLab_modele_fictif.csv", "text/csv", icon=":material/download:")
        uploaded = st.file_uploader("Fichier de cas", type=["csv", "xlsx"], max_upload_size=5)
        if uploaded:
            try:
                cases = parse_import(uploaded.getvalue(), uploaded.name)
                st.dataframe(case_frame(cases), hide_index=True, width="stretch", alt="Prévisualisation des hypothèses importées")
                choice = st.selectbox("Cas à utiliser", range(len(cases)), format_func=lambda i: cases[i]["nom"])
                confirm = st.checkbox("Remplacer le cas actif par le cas sélectionné", key="import_confirm")
                if st.button("Utiliser ce cas", disabled=not confirm, type="primary", icon=":material/check:"):
                    set_case(cases[choice])
                    st.session_state.page = "Tableau de bord"
                    st.rerun()
                if st.button("Comparer les lignes importées", icon=":material/compare_arrows:", disabled=len(cases)<2):
                    st.session_state.import_comparison = cases
                compared = st.session_state.get("import_comparison")
                if compared and compared == cases:
                    shown = cases[:20]
                    labels = [f"Cas {i+1}" for i in range(len(shown))]
                    fig = go.Figure(go.Bar(y=labels, x=[results(x)["pression_kPa"] for x in shown],
                        customdata=[x["nom"] for x in shown], orientation="h", marker_color=CYAN, name="Pression",
                        hovertemplate="%{customdata}<br>q : %{x:.2f} kPa<extra></extra>"))
                    fig.add_trace(go.Scatter(y=labels, x=[x["seuil_kPa"] for x in shown], mode="markers", name="Seuil fourni", marker=dict(color=PINK, symbol="diamond", size=11)))
                    fig.update_layout(xaxis_title="Pression (kPa)")
                    fig.update_yaxes(autorange="reversed")
                    chart(fig, "import_comparison", "Pressions et seuils des premières lignes importées, dans l'ordre du tableau", max(360, len(shown)*30+110))
                    if len(cases) > 20:
                        st.caption(f"Graphique : 20 premières lignes sur {len(cases)}. Le tableau et l'export contiennent tous les cas.")
                    st.download_button("Exporter les calculs importés", excel_bytes(case_frame(cases)), "FondationLab_import_calcule.xlsx", icon=":material/download:")
            except (ValueError, UnicodeError) as exc:
                st.error(str(exc))
    st.caption("Données conservées uniquement dans la session du navigateur. Un redémarrage du serveur ou une nouvelle session peut les effacer ; exporter les cas utiles.")


def reference_checks():
    other = {**EXAMPLE, "largeur_m": 2.0}
    zero = {**EXAMPLE, "charge_kN": 0.0}
    checks = [
        ("Surface de 2 × 1,5 m", results(EXAMPLE)["surface_m2"], 3.0),
        ("Pression de 300 kN sur 3 m²", results(EXAMPLE)["pression_kPa"], 100.0),
        ("Pression de 300 kN sur 4 m²", results(other)["pression_kPa"], 75.0),
        ("Charge nulle", results(zero)["pression_kPa"], 0.0),
        ("Triangle (240,300,360), seuil 110 kPa", exact_risk(EXAMPLE, 240, 300, 360)["probabilite"], .125),
        ("Charge fixe 330 kN, q = seuil", exact_risk(EXAMPLE, 330, 330, 330)["probabilite"], 0.0),
        ("Charge fixe 360 kN, q > seuil", exact_risk(EXAMPLE, 360, 360, 360)["probabilite"], 1.0),
    ]
    return pd.DataFrame([{"Vérification": label, "Calculé": got, "Attendu": expected,
                          "Résultat": "OK" if math.isclose(got, expected, abs_tol=1e-10) else "ÉCHEC"}
                         for label, got, expected in checks])


def learn_page():
    st.header("Comprendre et tester")
    tabs = st.tabs(["Le principe", "Tests guidés", "Formules et limites"])
    with tabs[0]:
        st.subheader("Même charge, surface plus grande")
        st.markdown("Une semelle est une base élargie sous un poteau. Dans ce modèle, la charge se répartit uniformément sur toute sa surface. Agrandir cette surface réduit la pression moyenne pour une même charge.")
        st.latex(r"A=L\times B \qquad q=\frac{N}{A}")
        st.markdown("**Unités :** longueur en m, surface en m², charge en kN, pression en kPa. Un kPa correspond à un kN/m². Le kN est une unité de force, pas une masse en kilogrammes.")
        st.subheader("Un seuil, pas un verdict")
        st.markdown("Le seuil de 110 kPa de l'exemple est fictif. Une pression inférieure à ce seuil signifie seulement que cette comparaison est satisfaite. L'application ne connaît ni le sol ni les autres conditions d'un ouvrage réel.")
        answer = st.radio("À charge constante, si la surface double, la pression moyenne…", ["est divisée par deux", "double", "ne change pas"], index=None)
        if st.button("Vérifier la réponse", icon=":material/check:"):
            if answer is None:
                st.info("Choisir une réponse.")
            elif answer == "est divisée par deux":
                st.success("Exact : q = N / A. Une surface doublée donne une pression divisée par deux.")
            else:
                st.info("Reprendre q = N / A : si le dénominateur double, le résultat est divisé par deux.")
    with tabs[1]:
        st.subheader("Un parcours de test complet")
        st.markdown("""
1. Dans la barre latérale, charger **l'exemple fictif**. Le tableau de bord doit afficher **3 m²**, **100 kPa**, et **90,91 %** du seuil fictif de 110 kPa.
2. Passer de **Coupe** à **Plan** : les dimensions restent 2 m et 1,5 m. Dans le formulaire, saisir B = 2 m puis calculer : **4 m²**, **75 kPa**. Recharger ensuite l'exemple.
3. Dans **Comparer**, conserver L = 2 m et B = 2 m pour la variante. Comparer : A = **100 kPa**, B = **75 kPa**, variation = **-25 %** et surface supplémentaire = **1 m²**.
4. Dans **Incertitudes**, utiliser **240 / 300 / 360 kN**, 20 000 tirages, graine 42. La probabilité théorique vaut **12,5 %** ; la simulation doit être proche, sans égalité imposée. La moyenne théorique vaut **100 kPa**, l'écart-type **8,165 kPa**, le quantile 95 % **113,675 kPa**.
5. Saisir **330 / 330 / 330 kN** : la pression vaut exactement **110 kPa**. Le dépassement est strict : **0 %**. Avec **360 / 360 / 360**, on obtient **120 kPa** et **100 %**.
6. Dans **Données et rapports**, exporter CSV et Excel, puis préparer et télécharger le PDF. Importer le modèle fictif : la prévisualisation doit afficher deux pressions, **100** et **75 kPa**. Rien ne remplace le cas actif sans confirmation.
7. Essayer un nom vide, une charge minimale supérieure à la maximale ou un fichier incomplet : un message doit expliquer l'erreur, sans interrompre l'application.
""")
        if st.button("Exécuter les vérifications de référence", icon=":material/fact_check:", type="primary"):
            st.dataframe(reference_checks(), hide_index=True, width="stretch", alt="Résultats des sept calculs de référence")
    with tabs[2]:
        st.subheader("Probabilités : ce qui varie vraiment")
        st.markdown("La charge est décrite par un minimum a, une valeur la plus probable m et un maximum b. La loi triangulaire privilégie les valeurs voisines de m. Ces trois nombres sont des hypothèses choisies, pas une loi ajustée automatiquement à des mesures.")
        st.latex(r"P(q>q_{ref})=P(N>q_{ref}LB)")
        st.latex(r"E[N]=\frac{a+m+b}{3}\qquad \mathrm{Var}(N)=\frac{a^2+m^2+b^2-am-ab-mb}{18}")
        st.markdown("**Test de 12,5 % :** 110 × 3 = 330 kN. Dans la partie descendante du triangle, P(N > 330) = (360 - 330)² / [(360 - 240)(360 - 300)] = 0,125.")
        st.markdown("**Quantile 95 % :** 95 % des pressions du modèle sont inférieures ou égales à cette valeur. **Graine :** nombre qui permet de reproduire les mêmes tirages. **IC à 95 % :** intervalle binomial exact de Clopper-Pearson pour l'estimation Monte-Carlo ; il ne couvre pas les erreurs de choix du modèle.")
        note(ASSUMPTIONS, rose=True)
        st.markdown("Références : [FHWA, Soils and Foundations](https://highways.dot.gov/sites/fhwa.dot.gov/files/FHWA-NHI-06-089.pdf) · [SciPy : loi triangulaire](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html) · [SciPy : intervalle binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)")


def main():
    st.set_page_config(page_title="FondationLab · Pression sous semelle", page_icon=":material/foundation:", layout="wide")
    for key, default in {"case": None, "revision": 0, "risk": None, "comparison": None, "page": "Tableau de bord"}.items():
        if key not in st.session_state:
            st.session_state[key] = default
    st.html(CSS)
    navigation = [
        ("Tableau de bord", "dashboard"), ("Comparer", "compare_arrows"),
        ("Incertitudes", "query_stats"), ("Données et rapports", "folder_open"),
        ("Comprendre et tester", "school"),
    ]
    with st.sidebar:
        st.html('<div class="fl-side-brand">FondationLab<span style="color:#78cfda">.</span></div><div class="fl-side-sub">MÉCANIQUE DES SOLS / ATELIER 01</div>')
        for label, icon in navigation:
            if st.button(label, icon=f":material/{icon}:", width="stretch", type="primary" if st.session_state.page == label else "secondary"):
                st.session_state.page = label
                st.rerun()
        st.space(30)
        with st.popover("Exemple fictif", icon=":material/science:", width="stretch"):
            st.markdown("**300 kN · 2 × 1,5 m · seuil fictif 110 kPa**")
            confirmed = st.checkbox("Remplacer le cas actif et ses analyses", value=st.session_state.case is None)
            if st.button("Charger l'exemple fictif", disabled=not confirmed, icon=":material/restart_alt:"):
                set_case(deepcopy(EXAMPLE))
                st.session_state.page = "Tableau de bord"
                st.rerun()
        st.caption("Modèle pédagogique · v" + VERSION)
    case = active_case()
    st.html('<div class="fl-top"><div><h1>FondationLab</h1><p>Pression sous semelle · charge, surface et incertitude</p></div><div class="fl-tag">GÉNIE CIVIL / LABORATOIRE PÉDAGOGIQUE</div></div>')
    origin = "Aperçu fictif · non enregistré" if st.session_state.case is None else case["origine"]
    st.html(f'<div class="fl-context"><span><b>{escape(case["nom"])}</b> / {escape(origin)}</span><span>Seuil fourni : <b>{fmt(case["seuil_kPa"])} kPa</b></span></div>')
    if st.session_state.page == "Tableau de bord":
        dashboard(case)
    elif st.session_state.page == "Comparer":
        comparison_page(case)
    elif st.session_state.page == "Incertitudes":
        risk_page(case)
    elif st.session_state.page == "Données et rapports":
        data_page(case)
    else:
        learn_page()
    st.html('<div class="fl-footer">Charge verticale centrée · contact uniforme idéalisé · aucune validation réglementaire ou professionnelle</div>')


if __name__ == "__main__":
    main()
