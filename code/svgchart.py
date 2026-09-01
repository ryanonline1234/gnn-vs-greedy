"""Minimal theme-aware inline-SVG charting. Colors are CSS custom properties so the
figures follow the page's light/dark tokens instead of baking in a palette."""
import math

def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

class Chart:
    def __init__(self, w, h, pad=(46, 18, 40, 58)):
        self.w, self.h = w, h
        self.pt, self.pr, self.pb, self.pl = pad
        self.parts = []
    @property
    def iw(self): return self.w - self.pl - self.pr
    @property
    def ih(self): return self.h - self.pt - self.pb
    def sx(self, v, lo, hi): return self.pl + (v - lo) / (hi - lo) * self.iw
    def sy(self, v, lo, hi): return self.pt + (1 - (v - lo) / (hi - lo)) * self.ih
    def add(self, s): self.parts.append(s)
    def frame(self):
        self.add(f'<rect x="{self.pl}" y="{self.pt}" width="{self.iw}" height="{self.ih}" '
                 f'fill="none" stroke="var(--rule)" stroke-width="1"/>')
    def hgrid(self, vals, lo, hi, fmt="{:.2f}"):
        for v in vals:
            y = self.sy(v, lo, hi)
            self.add(f'<line x1="{self.pl}" y1="{y:.1f}" x2="{self.pl+self.iw}" y2="{y:.1f}" '
                     f'stroke="var(--rule)" stroke-width="1" stroke-dasharray="2 4"/>')
            self.add(f'<text x="{self.pl-8}" y="{y+3.5:.1f}" text-anchor="end" '
                     f'class="ax">{_esc(fmt.format(v))}</text>')
    def xticks(self, vals, lo, hi, fmt="{:g}"):
        for v in vals:
            x = self.sx(v, lo, hi)
            self.add(f'<line x1="{x:.1f}" y1="{self.pt+self.ih}" x2="{x:.1f}" '
                     f'y2="{self.pt+self.ih+4}" stroke="var(--rule)" stroke-width="1"/>')
            self.add(f'<text x="{x:.1f}" y="{self.pt+self.ih+16}" text-anchor="middle" '
                     f'class="ax">{_esc(fmt.format(v))}</text>')
    def axlabel(self, xlab, ylab):
        self.add(f'<text x="{self.pl+self.iw/2}" y="{self.h-6}" text-anchor="middle" '
                 f'class="axl">{_esc(xlab)}</text>')
        self.add(f'<text transform="translate(13,{self.pt+self.ih/2}) rotate(-90)" '
                 f'text-anchor="middle" class="axl">{_esc(ylab)}</text>')
    def path(self, pts, color, width=2, dash=None, cls=None):
        d = " ".join(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        # cls="draw" pairs with pathLength=1 so CSS can run a dashoffset draw-in
        extra = f' class="{cls}" pathLength="1"' if cls else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{da}{extra}/>')
    def dot(self, x, y, color, r=4.2, title=None):
        t = f'<title>{_esc(title)}</title>' if title else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" '
                 f'stroke="var(--paper)" stroke-width="1.4">{t}</circle>')
    def text(self, x, y, s, cls="ann", anchor="start"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{_esc(s)}</text>')
    def render(self, title=""):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{_esc(title)}" '
                f'class="chart">' + "".join(self.parts) + "</svg>")
