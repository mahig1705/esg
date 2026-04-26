"""api/pdf_styles.py — shared colors, fonts, styles for ESGLens PDF."""
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm

# ── Brand palette ──────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0A1628")
TEAL   = colors.HexColor("#00D4AA")
AMBER  = colors.HexColor("#F59E0B")
RED    = colors.HexColor("#EF4444")
GREEN  = colors.HexColor("#10B981")
GREY   = colors.HexColor("#94A3B8")
LGREY  = colors.HexColor("#1E2D40")
WHITE  = colors.white
BLACK  = colors.black

RISK_COLOR = {"HIGH": RED, "CRITICAL": RED, "MODERATE": AMBER, "MEDIUM": AMBER, "LOW": GREEN}

# ── Styles ─────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def make_styles():
    s = {}
    s["h1"] = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=22, textColor=WHITE, spaceAfter=6)
    s["h2"] = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, textColor=TEAL, spaceAfter=4, spaceBefore=10)
    s["h3"] = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11, textColor=WHITE, spaceAfter=3, spaceBefore=6)
    s["body"] = ParagraphStyle("body", fontName="Helvetica", fontSize=9, textColor=GREY, spaceAfter=4, leading=14)
    s["label"] = ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=7, textColor=TEAL, spaceAfter=2)
    s["mono"] = ParagraphStyle("mono", fontName="Courier", fontSize=8, textColor=WHITE, spaceAfter=2)
    s["footer"] = ParagraphStyle("footer", fontName="Helvetica", fontSize=7, textColor=GREY)
    s["warning"] = ParagraphStyle("warning", fontName="Helvetica-Bold", fontSize=8, textColor=AMBER, spaceAfter=3)
    return s

STYLES = make_styles()
