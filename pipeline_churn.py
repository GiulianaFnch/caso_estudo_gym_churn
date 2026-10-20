"""
Caso de estudo — Previsão de churn de clientes de um ginásio
Projeto de Inteligência Artificial · CTeSP DAIA · IPGT · 2026/2027
Autores: Alisson Alencar, Giuliana Finochio

Pipeline reprodutível (pandas + scikit-learn):
dados brutos → limpeza → divisão treino/teste → pré-processamento → modelo → avaliação → modelo guardado

Executar:  python pipeline_churn.py
Corre do princípio ao fim sem intervenção manual.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # gera figuras sem janela (corre em qualquer máquina)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 0. Configuração
# ---------------------------------------------------------------------------
# Em macOS com numpy 2.x (Accelerate) surgem avisos falsos "encountered in matmul";
# os resultados são idênticos aos obtidos no Colab/Linux, por isso silenciamos só esses.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "gym_churn_us.csv"
MODEL_PATH = BASE_DIR / "models" / "pipeline_churn.joblib"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)

# Grupos de variáveis (definidos na fase de Compreensão dos Dados)
BINARY_FEATURES = ["gender", "Near_Location", "Partner", "Promo_friends", "Phone", "Group_visits"]
CATEGORICAL_FEATURES = ["Contract_period"]  # 1, 6 ou 12 meses → tratada como categoria
NUMERIC_FEATURES = [
    "Age",
    "Avg_additional_charges_total",
    "Month_to_end_contract",
    "Lifetime",
    "Avg_class_frequency_total",
    "Avg_class_frequency_current_month",
]
# Variáveis sem valor preditivo para o negócio (identificadores). O dataset não tem
# nenhuma, mas a lista fica preparada caso uma versão futura do CSV inclua IDs.
IRRELEVANT_COLUMNS = ["customer_id", "id", "ID"]


# ---------------------------------------------------------------------------
# 1. Carregar os dados (o CSV original nunca é alterado: só leitura)
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[1] Dados carregados: {df.shape[0]} linhas × {df.shape[1]} colunas")
    return df


# ---------------------------------------------------------------------------
# 2. Limpeza (operações que NÃO aprendem nada com os dados → sem fuga de informação)
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in IRRELEVANT_COLUMNS if c in df.columns])
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])  # sem alvo não há como treinar/avaliar
    print(
        f"[2] Limpeza: {n_dup} duplicados removidos; "
        f"valores omissos nas variáveis explicativas = {int(df.drop(columns=TARGET).isna().sum().sum())} "
        "(tratados por imputação dentro do pipeline)"
    )
    return df


# ---------------------------------------------------------------------------
# 3. Pipeline: ColumnTransformer (imputação, codificação, normalização) + modelo
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    numeric = Pipeline(
        [
            ("imputacao", SimpleImputer(strategy="median")),
            ("normalizacao", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputacao", SimpleImputer(strategy="most_frequent")),
            ("codificacao", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    binary = Pipeline([("imputacao", SimpleImputer(strategy="most_frequent"))])

    preprocess = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin", binary, BINARY_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessamento", preprocess),
            ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def evaluate(name: str, y_true, y_pred, y_proba=None) -> dict:
    m = {
        "modelo": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "matriz_confusao": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_proba)
    return m


def main() -> None:
    # 1–2. Carregar e limpar
    df = clean_data(load_data(DATA_PATH))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES]
    y = df[TARGET]
    print(f"    Taxa de churn global: {y.mean():.1%}")

    # 3. Divisão treino/teste ANTES de qualquer transformação aprendida (evita data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"[3] Divisão 80/20 estratificada: treino={len(X_train)} (churn {y_train.mean():.1%}), "
        f"teste={len(X_test)} (churn {y_test.mean():.1%})"
    )

    # 4. Baseline: prevê sempre a classe maioritária (cliente fica)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    m_base = evaluate("Baseline (DummyClassifier)", y_test, baseline.predict(X_test))

    # 5. Modelo: o pipeline só aprende (fit) com o treino
    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    print(f"[4] Validação cruzada (5 folds, só treino): F1 = {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    m_lr = evaluate("Regressão Logística", y_test, y_pred, y_proba)
    m_lr["cv_f1_media"] = cv_f1.mean()
    m_lr["cv_f1_desvio"] = cv_f1.std()

    # 6. Avaliação no conjunto de teste
    print("\n[5] Avaliação no conjunto de teste")
    print(pd.DataFrame([m_base, m_lr]).drop(columns="matriz_confusao").round(3).to_string(index=False))
    print("\nMatriz de confusão (Regressão Logística) [linhas=real, colunas=previsto]:")
    print(np.array(m_lr["matriz_confusao"]))
    print("\n" + classification_report(y_test, y_pred, target_names=["Fica (0)", "Churn (1)"], digits=3))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (title, pred) in zip(
        axes, [("Baseline (DummyClassifier)", baseline.predict(X_test)), ("Regressão Logística", y_pred)]
    ):
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Fica", "Churn"], cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "matriz_confusao.png", dpi=200)
    plt.close(fig)

    # 7. Interpretação: coeficientes (variáveis normalizadas → comparáveis entre si)
    names = pipe.named_steps["preprocessamento"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    coefs = pd.DataFrame(
        {"variavel": names, "coeficiente": pipe.named_steps["modelo"].coef_[0]}
    )
    coefs["odds_ratio"] = np.exp(coefs["coeficiente"])
    coefs["abs"] = coefs["coeficiente"].abs()
    coefs = coefs.sort_values("abs", ascending=False).drop(columns="abs")
    coefs.to_csv(OUT_DIR / "coeficientes.csv", index=False)
    print("[6] Coeficientes ordenados por influência (|coef|):")
    print(coefs.round(3).to_string(index=False))

    top = coefs.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["variavel"], top["coeficiente"], color=["#c0392b" if c > 0 else "#2471a3" for c in top["coeficiente"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coeficiente (positivo = aumenta o risco de churn)")
    ax.set_title("Top 10 coeficientes — Regressão Logística")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coeficientes.png", dpi=200)
    plt.close(fig)

    # 8. Guardar o pipeline completo (pré-processamento + modelo) e verificar que recarrega
    joblib.dump(pipe, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    assert (reloaded.predict(X_test) == y_pred).all(), "O pipeline recarregado não reproduz as previsões"
    print(f"\n[7] Pipeline guardado em {MODEL_PATH.relative_to(BASE_DIR)} e verificado após recarregar")

    with open(OUT_DIR / "metricas.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": m_base, "regressao_logistica": m_lr}, f, indent=2, ensure_ascii=False)
    print(f"    Métricas guardadas em outputs/metricas.json")


if __name__ == "__main__":
    main()
