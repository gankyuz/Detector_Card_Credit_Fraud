# Detector de Fraudes em Cartão de Crédito

## Motivação

Fraudes em cartões de crédito causam prejuízos bilionários globalmente. Esse projeto nasceu do meu interesse em entender como algoritmos de ML se comportam diante de um problema real, desequilibrado e com alto custo de erro, porque num cenário de fraude, deixar uma transação suspeita passar é muito pior do que gerar um falso alarme.

Mais do que aplicar modelos prontos, quis investigar como as escolhas de pré-processamento e balanceamento impactam diretamente os resultados, e o que as métricas realmente dizem quando a acurácia mente.

## Dataset


* Fonte: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* 284.807 transações, sendo apenas 492 fraudes (< 0,2%)
* Features anonimizadas via PCA (V1–V28) + Amount e Time
* Target: Class → 0 = legítima, 1 = fraude

> O desbalanceamento extremo é o principal desafio do projeto. Modelos treinados sem tratamento atingem ~99% de acurácia simplesmente ignorando as fraudes, o que chamamos de acurácia ilusória.

## Tecnologias


* Python 3
* Pandas & NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

## Pipeline do Projeto

 
### 1. Análise Exploratória
- Distribuição das classes
- Histogramas de todas as features
- Estatísticas descritivas com `df.describe()`
### 2. Pré-processamento
- **`Amount`** → normalizado com `RobustScaler` (resistente a outliers extremos de valor)
- **`Time`** → reescalonado para o intervalo [0, 1]
- Dados embaralhados com seed fixo para reprodutibilidade
- Divisão em **treino (240k) / teste (22k) / validação (~23k)**
### 3. Experimentação em duas etapas
 
O ponto central do projeto foi rodar os experimentos **duas vezes**:
 
| Etapa | Dados | Objetivo |
|---|---|---|
| 1ª | Desbalanceados | Entender o comportamento base dos modelos |
| 2ª | Balanceados | Medir o impacto real do balanceamento nas métricas |
 
## Modelos Treinados

 
### Regressão Logística
Modelo baseline. Com dados desbalanceados, a acurácia parecia perfeita — mas o `classification_report` revelou que o modelo simplesmente não detectava fraudes.
 
> **Conclusão:** confirma que acurácia não é a métrica certa para esse problema.
 
---
 
### Rede Neural Rasa (Keras)
```
InputLayer → Dense(2, relu) → BatchNormalization → Dense(1, sigmoid)
```
Treinada por 5 épocas com `ModelCheckpoint` para salvar o melhor modelo. Apresentou **recall maior para fraudes**, captura mais casos suspeitos, ainda que com mais falsos positivos.
 
---
 
### Random Forest
Modelo ensemble robusto. Boa interpretabilidade via importância de features, útil para etapas futuras de análise.
 
---
 
### Gradient Boosting
Ensemble sequencial com aprendizado iterativo sobre os erros. Testado com `n_estimators=50`.
 
---

### LinearSVC com `class_weight='balanced'`
Abordagem direta ao desbalanceamento: o parâmetro `class_weight='balanced'` faz o algoritmo penalizar mais os erros na classe minoritária automaticamente, sem alterar o dataset.

## Métricas utilizadas


Para dados desbalanceados, **acurácia engana**. As métricas relevantes são:
 
- **Precision:** dos que o modelo acusou como fraude, quantos eram de fato fraudes?
- **Recall:** de todas as fraudes reais, quantas o modelo detectou?
- **F1-score:** média harmônica entre precision e recall
  
A escolha do modelo ideal depende do objetivo de negócio:
- Prioridade em **não deixar fraudes passarem** → maximizar recall
- Prioridade em **minimizar falsos alarmes** → maximizar precision


## Principais Aprendizados
 
- Desbalanceamento de classes é um problema estrutural, não um detalhe
- O `RobustScaler` é uma escolha mais defensável que o `StandardScaler` quando há outliers fortes
- Separar treino, teste e validação é essencial para uma avaliação honesta
- Comparar modelos com e sem balanceamento revela muito mais do que testar apenas uma configuração

## Sobre
 
Projeto desenvolvido como parte da minha jornada em Ciência da Computação, explorando Machine Learning aplicado a problemas reais.  
> Gerou resultados visivelmente diferentes da versão sem esse ajuste.
 
---
