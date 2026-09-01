# backend/report.py
"""PDF report generation for the health dashboard.

Uses fpdf2 (pure-Python) to create a clean PDF summarising the patient
profile, calculated metrics and disease predictions.

All text is sanitised to latin-1 before being written so fpdf2 never
throws FPDFUnicodeEncodingException.  Arial is used throughout (not
Helvetica) because it is embedded as a core font and supports the full
latin-1 character set.
"""

from fpdf import FPDF
import os


# ── latin-1 sanitiser ────────────────────────────────────────────────
def _safe(text: str) -> str:
    """Replace common Unicode characters and encode to latin-1."""
    return (
        str(text)
        .replace("\u2014", "-")   # em dash
        .replace("\u2013", "-")   # en dash
        .replace("\u2018", "'")   # left single quote
        .replace("\u2019", "'")   # right single quote
        .replace("\u201c", '"')   # left double quote
        .replace("\u201d", '"')   # right double quote
        .replace("\u2022", "*")   # bullet
        .replace("\u2026", "...")  # ellipsis
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


class PDFReport(FPDF):
    """Custom FPDF subclass with header, footer and section helpers."""

    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, _safe("Smart Health Dashboard - Health Report"),
                  ln=True, align="C")
        self.set_font("Arial", size=10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 7,
                  _safe("AI-Powered Personalised Nutrition & Disease Risk Analysis"),
                  ln=True, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(3)
        # Divider line
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-13)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, _safe(f"Page {self.page_no()}  |  Smart Health Dashboard"),
                  align="C")
        self.set_text_color(0, 0, 0)

    def section_title(self, title: str):
        """Render a bold section heading with a light background."""
        self.set_font("Arial", "B", 12)
        self.set_fill_color(241, 245, 249)
        self.cell(0, 9, _safe(title), ln=True, fill=True)
        self.ln(1)

    def key_value(self, key: str, value):
        """Render a key: value row."""
        self.set_font("Arial", "B", 10)
        self.cell(55, 7, _safe(f"{key}:"), border=0)
        self.set_font("Arial", size=10)
        # Use multi_cell so long values wrap properly
        x_after = self.get_x()
        self.multi_cell(0, 7, _safe(str(value)))

    def multiline_list(self, title: str, items: list):
        """Render a titled bullet list."""
        self.set_font("Arial", "B", 10)
        self.cell(0, 8, _safe(title), ln=True)
        self.set_font("Arial", size=10)
        for item in items:
            # Guard against page overflow
            if self.get_y() > self.h - self.b_margin - 15:
                self.add_page()
            if isinstance(item, dict):
                food = item.get("food", item.get("Dish Name", str(item)))
                reason = item.get("reason", "")
                line = f"  - {food}"
                if reason:
                    line += f": {reason}"
            else:
                line = f"  - {item}"
            self.multi_cell(0, 6, _safe(line))
        self.ln(2)


def generate_pdf(report_data: dict, output_path: str) -> None:
    """Write a health report PDF to *output_path*."""
    pdf = PDFReport()
    pdf.set_margins(left=10, top=20, right=10)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Personal Details ─────────────────────────────────────────────
    pdf.section_title("Personal Details")
    for k, v in report_data.get("personal", {}).items():
        pdf.key_value(k.replace("_", " ").capitalize(), v)
    pdf.ln(3)

    # ── Health Metrics ───────────────────────────────────────────────
    pdf.section_title("Health Metrics")
    for k, v in report_data.get("metrics", {}).items():
        if v is not None:
            pdf.key_value(k, v)
    pdf.ln(3)

    # ── Disease Predictions ──────────────────────────────────────────
    pdf.section_title("Disease Predictions")
    for k, v in report_data.get("predictions", {}).items():
        pdf.key_value(k, v)
    pdf.ln(3)

    # ── Recommendations ──────────────────────────────────────────────
    recs = report_data.get("recommendations", {})

    tips = recs.get("nutrition_tips", [])
    if tips:
        pdf.section_title("Nutrition Tips")
        pdf.multiline_list("", tips)

    avoid = recs.get("foods_to_avoid", [])
    if avoid:
        pdf.section_title("Foods to Avoid")
        pdf.multiline_list("", avoid)

    # ── Write file ───────────────────────────────────────────────────
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pdf.output(output_path)


def generate_and_save(report_data: dict,
                      filename: str = "health_report.pdf") -> str:
    """Convenience wrapper — writes to reports/ and returns the path."""
    reports_dir = os.path.join(os.path.abspath(os.curdir), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, filename)
    generate_pdf(report_data, out_path)
    return out_path
