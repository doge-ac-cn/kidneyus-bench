#!/usr/bin/env python3
"""Amendment 1 section 4.3 domain-specific augmentations (a)-(f).

Composite op: with per-image probability 0.8, one or more of the six ops is
sampled uniformly and applied. Pure image-array math (no dataset access) so
it can be unit-tested before the OSF lock without running any Arm computation.
"""
import numpy as np


def _rand_side(rng, lo, hi):
    return rng.uniform(lo, hi)


def console_text(img, rng):
    """(a) 1-4 synthetic text lines, digits/latin, font height 10-40 px,
    opacity 0.3-0.9, uniform placement. Pure-numpy bitmap font (no PIL font
    dependency): 5x7 glyphs scaled to the target height."""
    h, w = img.shape[:2]
    glyphs = {c: G for c, G in _GLYPHS.items()}
    out = img.astype(np.float32).copy()
    n_lines = rng.integers(1, 5)
    for _ in range(n_lines):
        text = "".join(rng.choice(list(glyphs.keys()), size=rng.integers(6, 14)))
        gh = int(_rand_side(rng, 10, 40))
        gw = 6 * gh // 7
        alpha = _rand_side(rng, 0.3, 0.9)
        x0 = int(rng.integers(0, max(w - len(text) * gw, 1)))
        y0 = int(rng.integers(0, max(h - gh, 1)))
        canvas = np.zeros((gh, gw * len(text)), dtype=np.float32)
        for i, ch in enumerate(text):
            g = np.kron(glyphs[ch], np.ones((max(gh // 7, 1), max(gw // 5, 1))))
            g = np.kron(_GLYPHS[ch], np.ones((gh // 7 or 1, gw // 5 or 1)))[:gh, :gw]
            canvas[: g.shape[0], i * gw: i * gw + g.shape[1]] = g[:gh, :gw]
        m = canvas > 0
        region = out[y0:y0 + gh, x0:x0 + canvas.shape[1]]
        if region.shape[:2] != canvas.shape:
            continue
        out[y0:y0 + gh, x0:x0 + canvas.shape[1]][m] = (
            (1 - alpha) * region[m] + alpha * 255.0)
    return out


_GLYPHS = {c: np.array([[int(b) for b in row] for row in g], dtype=np.float32)
           for c, g in {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "7": ["11111", "00001", "00010", "00010", "00100", "00100", "00100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
    "H": ["10001", "10001", "11111", "10001", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "S": ["01111", "10000", "01110", "00001", "00001", "10001", "01110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
}.items()}


def fan_vignette(img, rng):
    """(b) sector-boundary darkening up to 40%."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - h * 0.55, xx - w * 0.5)
    edge = rng.uniform(-np.pi, np.pi)
    width = _rand_side(rng, 0.15, 0.6)
    d = np.abs(np.angle(np.exp(1j * (ang - edge))))
    strength = _rand_side(rng, 0.2, 0.4)
    mask = 1.0 - strength * np.exp(-(d / width) ** 2)
    return img.astype(np.float32) * mask


def gamma_contrast(img, rng):
    """(c) gamma in [0.6, 1.6]; contrast +/-30%."""
    x = img.astype(np.float32) / 255.0
    g = _rand_side(rng, 0.6, 1.6)
    x = np.power(np.clip(x, 0, 1), g)
    c = _rand_side(rng, 0.7, 1.3)
    x = np.clip((x - 0.5) * c + 0.5, 0, 1)
    return x * 255.0


def vertical_shadow(img, rng):
    """(d) vertical shadow gradient, strength <= 0.4."""
    h = img.shape[0]
    strength = _rand_side(rng, 0.2, 0.4)
    top_down = rng.integers(0, 2)  # shadow from top or bottom
    ramp = np.linspace(strength, 0.0, h) if top_down else np.linspace(0.0, strength, h)
    return img.astype(np.float32) * ramp[:, None, None] if img.ndim == 3 \
        else img.astype(np.float32) * ramp[:, None]


def speckle(img, rng):
    """(e) multiplicative speckle noise, sigma <= 0.15."""
    sigma = _rand_side(rng, 0.05, 0.15)
    x = img.astype(np.float32)
    noise = 1.0 + rng.normal(0.0, sigma, x.shape[:2])[..., None] if x.ndim == 3 \
        else 1.0 + rng.normal(0.0, sigma, x.shape)
    return x * noise


def cutout(img, rng):
    """(f) 1-3 rectangles, each <= 10% of image area."""
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]
    for _ in range(rng.integers(1, 4)):
        area = _rand_side(rng, 0.02, 0.10)
        ah = int(h * np.sqrt(area * _rand_side(rng, 0.4, 2.5)))
        aw = int(max(area * h * w / max(ah, 1), 4))
        y0 = int(rng.integers(0, max(h - ah, 1)))
        x0 = int(rng.integers(0, max(w - aw, 1)))
        out[y0:y0 + ah, x0:x0 + aw] = 0.0
    return out


OPS = {"text": console_text, "fan": fan_vignette, "gamma": gamma_contrast,
       "shadow": vertical_shadow, "speckle": speckle, "cutout": cutout}


def domain_augment(img_u8, rng, p=0.8):
    """Composite: with prob p, apply one or more ops sampled uniformly."""
    x = img_u8.astype(np.float32)
    if rng.random() < p:
        names = rng.choice(list(OPS.keys()), size=int(rng.integers(1, len(OPS) + 1)),
                           replace=False)
        for name in names:
            x = OPS[name](x, rng)
    return np.clip(x, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    synthetic = (rng.random((480, 640)) * 255).astype(np.uint8)
    out = domain_augment(synthetic, rng)
    assert out.shape == synthetic.shape and out.dtype == np.uint8
    for name, op in OPS.items():
        o = op(synthetic, rng)
        assert o.shape == synthetic.shape, name
    print("aug ops unit test ok (synthetic arrays only, no dataset access)")
