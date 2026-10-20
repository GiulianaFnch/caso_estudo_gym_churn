"""Gera docs/diagrama_pipeline.png — fluxo do pipeline com os pontos de controlo de fuga de informação."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "diagrama_pipeline.png"

steps = [
    ("Dados brutos", "gym_churn_us.csv\n(só leitura)", "#d6eaf8"),
    ("Limpeza", "remover IDs e\nduplicados", "#d6eaf8"),
    ("Divisão\ntreino/teste", "80/20 · stratify\nrandom_state=42", "#fdebd0"),
    ("Pré-\nprocessamento", "ColumnTransformer:\nimputação, one-hot,\nnormalização", "#d5f5e3"),
    ("Modelo", "Regressão\nLogística", "#d5f5e3"),
    ("Avaliação", "matriz de confusão,\nprecision, recall, F1\nvs. baseline", "#e8daef"),
    ("Modelo guardado", "pipeline_churn\n.joblib", "#e8daef"),
]

fig, ax = plt.subplots(figsize=(14, 4.6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.6)
ax.axis("off")

w, h, gap, y = 1.7, 1.5, 0.27, 1.9
xs = [0.2 + i * (w + gap) for i in range(len(steps))]
for x, (title, sub, color) in zip(xs, steps):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#34495e", lw=1.2))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x + w / 2, y + 0.5, sub, ha="center", va="center", fontsize=8)
for x in xs[:-1]:
    ax.annotate("", xy=(x + w + gap, y + h / 2), xytext=(x + w, y + h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=1.4))

# Zona "Pipeline do scikit-learn" (fit só no treino)
px0, px1 = xs[3] - 0.1, xs[4] + w + 0.1
ax.add_patch(FancyBboxPatch((px0, y - 0.25), px1 - px0, h + 0.5, boxstyle="round,pad=0.02",
                            fc="none", ec="#1e8449", lw=1.6, ls="--"))
ax.text((px0 + px1) / 2, y + h + 0.4, "sklearn Pipeline → .fit() só com dados de TREINO",
        ha="center", fontsize=9, color="#1e8449", weight="bold")

# Notas de fuga de informação
notes = [
    (xs[1] + w / 2, "① Limpeza sem\nestatísticas aprendidas"),
    (xs[2] + w / 2, "② Divisão ANTES\nde aprender\ntransformações"),
    ((xs[3] + xs[4] + w) / 2, "③ Medianas, modas e médias/desvios\ncalculados só no treino e\naplicados ao teste (transform)"),
    (xs[5] + w / 2, "④ Teste usado uma única vez,\nno fim"),
]
for x, txt in notes:
    ax.text(x, 0.75, txt, ha="center", va="center", fontsize=8, color="#922b21",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdedec", ec="#922b21", lw=0.8))
    ax.plot([x, x], [1.15, y - 0.27], color="#922b21", lw=0.8, ls=":")

ax.text(7, 0.08, "Legenda: a vermelho, os pontos onde se evita a fuga de informação (data leakage)",
        ha="center", fontsize=8, style="italic", color="#555")

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Diagrama guardado em {OUT}")
