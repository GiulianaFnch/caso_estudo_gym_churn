# Previsão de Churn de Clientes de um Ginásio — Ficha de Projeto e CRISP-DM

**Projeto de Inteligência Artificial · CTeSP DAIA · IPGT · 2026/2027**
Alisson Alencar · Giuliana Finochio — Docente: Jorge Pereira Duque

> Documento de trabalho com o conteúdo do relatório. Os números vêm de `outputs/metricas.json` e
> `outputs/coeficientes.csv` (gerados por `pipeline_churn.py`). O relatório final segue o modelo
> `Template Trabalhos.doc` (Times New Roman 12, espaçamento 1,5, máx. 6 páginas).

---

## 1. Ficha de projeto

| Campo | Definição |
|---|---|
| **Problema de negócio** | Os ginásios perdem receita quando os clientes cancelam a subscrição (*churn*). Conquistar um cliente novo custa mais do que reter um atual, por isso interessa identificar com antecedência quem está em risco. |
| **Utilizador** | Gestor/equipa comercial e de retenção do ginásio. |
| **Decisão apoiada** | Que clientes contactar este mês com uma ação de retenção (telefonema, aula de grupo gratuita, desconto na renovação). |
| **Variável-alvo** | `Churn` (1 = o cliente cancelou no mês seguinte; 0 = manteve-se). Problema de **classificação binária**. |
| **Dados** | *Gym customers features and churn* (Kaggle, Vinueza), ficheiro `gym_churn_us.csv`: 4000 clientes, 13 variáveis explicativas, sem valores omissos, 26,5 % de churn. |
| **Critério de sucesso** | No conjunto de teste: **recall da classe churn ≥ 0,75** (apanhar pelo menos 3 em cada 4 clientes que vão sair) e **F1 claramente superior ao baseline** (DummyClassifier). |
| **Fora do âmbito** | Desenhar ou testar as campanhas de retenção; prever *quando* o cliente sai (análise de sobrevivência); otimizar outros modelos (árvores, *boosting*); colocar o modelo em produção/aplicação web; dados pessoais identificáveis. |

## 2. Papéis da equipa (proposta — confirmar)

| Responsabilidade | Principal | Apoio |
|---|---|---|
| Processamento de dados e código Python (pipeline) | Alisson | Giuliana (revisão) |
| Treino do modelo e extração de métricas | Alisson | Giuliana |
| Redação e formatação do relatório (Resumo, Introdução, CRISP-DM, Conclusões) | Giuliana | Alisson (revisão técnica) |
| Gestão de ficheiros/repositório, README, requirements, apresentação de 7 min | Giuliana | Alisson |

## 3. Plano em 3 mini-sprints (2 semanas)

| Sprint | Atividades | Entregável |
|---|---|---|
| **1 — Planeamento e desenho** (dias 1–4) | Ficha de projeto, riscos, diagrama do pipeline, CRISP-DM fases 1–2 | Ficha de projeto, `docs/diagrama_pipeline.png`, estrutura do relatório |
| **2 — Núcleo funcional** (dias 5–9) | Carregar CSV, limpeza, divisão 80/20 estratificada, `Pipeline` + `ColumnTransformer`, Regressão Logística, baseline; CRISP-DM fases 3–4 | `pipeline_churn.py` / `caso_estudo.ipynb` reprodutível |
| **3 — Integração, validação e apresentação** (dias 10–14) | Métricas, comparação com baseline, 3 coeficientes, `joblib`, teste de execução completa, README, requirements, CRISP-DM fases 5–6, declaração de uso de IA, slides | Relatório PDF, código, `.joblib`, README, requirements.txt, apresentação |

## 4. Riscos principais e mitigação

| # | Risco | Mitigação |
|---|---|---|
| R1 | **Fuga de informação (*data leakage*)**: o pré-processamento aprender estatísticas (médias, medianas) com dados de teste, inflacionando as métricas. | Divisão treino/teste antes de qualquer transformação; todas as transformações dentro de um `Pipeline` do scikit-learn, ajustado (`fit`) só no treino; teste usado uma única vez no fim. |
| R2 | **Desequilíbrio de classes** (26,5 % churn): um modelo pode ter boa *accuracy* sem detetar nenhum cliente em risco. | `stratify=y` na divisão; avaliar com precision, recall e F1 da classe churn (não accuracy); comparar com o DummyClassifier, que mostra que accuracy ≈ 73,5 % não vale nada. |
| R3 | **Variáveis muito correlacionadas** (frequência total vs. mês atual, r = 0,95; duração do contrato vs. meses até ao fim, r = 0,97) tornam os coeficientes difíceis de interpretar isoladamente. | Normalizar as variáveis (coeficientes comparáveis); interpretar em conjunto as variáveis correlacionadas; validação cruzada para confirmar estabilidade (F1 = 0,858 ± 0,017). |

*Risco adicional para discutir:* `Avg_class_frequency_current_month` e `Month_to_end_contract` descrevem o mês atual. São legítimas **se** forem conhecidas no momento da previsão (início do mês para o qual se prevê o churn) — é o pressuposto assumido.

## 5. Representação do pipeline

Ver `docs/diagrama_pipeline.png`:
dados brutos → limpeza → divisão treino/teste → pré-processamento → modelo → avaliação → modelo guardado.
Pontos de controlo da fuga de informação: ① limpeza sem estatísticas aprendidas; ② divisão antes de aprender transformações; ③ imputação/normalização ajustadas só no treino (dentro do `Pipeline`); ④ teste usado uma única vez.

---

## 6. CRISP-DM (um parágrafo por fase — o que foi feito e a decisão tomada)

**6.1 Compreensão do negócio.** O ginásio pretende reduzir a perda de clientes. Definiu-se como objetivo prever, para cada cliente, a probabilidade de cancelar no mês seguinte, para que a equipa de retenção possa priorizar contactos. *Decisão:* tratar o problema como classificação binária e privilegiar o **recall** da classe churn, porque deixar escapar um cliente que vai sair (falso negativo) custa mais do que contactar um cliente que ficaria (falso positivo).

**6.2 Compreensão dos dados.** O dataset tem 4000 registos e 14 colunas numéricas: 6 binárias (género, proximidade, empresa parceira, promoção "traz um amigo", telefone, aulas de grupo), a duração do contrato (1, 6 ou 12 meses), 6 contínuas (idade, gastos adicionais, meses até ao fim do contrato, antiguidade, frequência média total e do mês atual) e o alvo `Churn` (26,5 % positivos). Não há valores omissos, duplicados nem identificadores. As variáveis mais associadas ao churn são a antiguidade (r = −0,44), a frequência no mês atual (−0,41), a idade (−0,41) e a duração do contrato (−0,39); género e telefone praticamente não se relacionam com o alvo. *Decisão:* manter todas as variáveis (não há IDs a retirar) e tratar `Contract_period` como categórica.

**6.3 Preparação dos dados.** O CSV é lido sem ser alterado. A limpeza (remover IDs e duplicados) não aprende nada com os dados. Os dados foram divididos em treino (3200) e teste (800), com `random_state=42` e estratificação, mantendo 26,5 % de churn em ambos. O pré-processamento foi construído com um `ColumnTransformer`: imputação pela mediana + `StandardScaler` nas contínuas, imputação pela moda + `OneHotEncoder` na duração do contrato, imputação pela moda nas binárias. *Decisão:* incluir a imputação mesmo sem omissos, para que o pipeline guardado seja robusto a dados novos incompletos.

**6.4 Modelação.** Juntou-se ao pré-processamento um único modelo — **Regressão Logística** (`max_iter=1000`) — num `Pipeline` do scikit-learn, e como termo de comparação um `DummyClassifier` que prevê sempre a classe maioritária. A validação cruzada estratificada em 5 *folds* sobre o treino deu F1 = 0,858 ± 0,017, mostrando que o modelo é estável. *Decisão:* manter os hiperparâmetros por omissão e o limiar de 0,5, dado que o critério de sucesso já é cumprido e o enunciado privilegia a simplicidade.

**6.5 Avaliação.** No conjunto de teste:

| Modelo | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (DummyClassifier) | 0,000 | 0,000 | 0,000 |
| **Regressão Logística** | **0,884** | **0,825** | **0,854** |

Matriz de confusão da Regressão Logística (linhas = real, colunas = previsto): 565 verdadeiros negativos, 23 falsos positivos, 37 falsos negativos e 175 verdadeiros positivos (ROC-AUC = 0,977). O modelo deteta 82,5 % dos clientes que saem e acerta em 88,4 % dos alertas. O baseline obtém 73,5 % de *accuracy* sem detetar nenhum churn, o que confirma que a accuracy não serve para este problema. *Decisão:* o critério de sucesso (recall ≥ 0,75 e F1 muito acima do baseline) foi cumprido.

**Os 3 coeficientes mais influentes** (variáveis normalizadas; valor por +1 desvio-padrão):
1. `Avg_class_frequency_current_month` = **−4,22** — quanto mais o cliente vai ao ginásio no mês atual, menor o risco de sair.
2. `Lifetime` = **−3,76** — clientes com mais meses de casa são muito mais fiéis; os primeiros meses são o período crítico.
3. `Avg_class_frequency_total` = **+3,14** — tem de ser lido em conjunto com o 1.º (r = 0,95): para a mesma frequência atual, um histórico de frequência mais alto aumenta o risco. Ou seja, o sinal de alerta é a **queda da frequência** em relação ao hábito do cliente.

**6.6 Implementação.** O pipeline completo (pré-processamento + modelo) foi guardado com `joblib` em `models/pipeline_churn.joblib`; o script recarrega-o e verifica que reproduz as mesmas previsões. Todo o processo corre com `python pipeline_churn.py`, sem intervenção manual. Na prática, o ginásio poderia correr o modelo no início de cada mês sobre a base de clientes e entregar à equipa de retenção uma lista ordenada por probabilidade de churn. *Decisão:* como trabalho futuro, monitorizar o desempenho mensalmente e retreinar quando o recall descer abaixo de 0,75.

---

## 7. Uso de IA generativa (declaração)

Foi utilizada IA generativa (Claude, da Anthropic, e Gemini, da Google) para apoiar a estruturação do plano, a escrita de código e a revisão de texto. Todo o código foi executado e os resultados verificados pela equipa; as métricas apresentadas vêm da execução real do script. *(Completar com exemplos concretos de algo que a IA sugeriu e foi corrigido/validado pela equipa.)*
