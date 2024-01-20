# Análise de Modelos de Regressão para Previsão de Precipitação 🌧️
## Contexto 🌐

Este relatório conduz uma análise detalhada dos modelos de regressão aplicados à previsão da quantidade total de precipitação em Campo Grande. Exploraremos métricas, visualizações e comparações entre modelos, proporcionando uma compreensão abrangente das técnicas empregadas. 

## Métricas de Avaliação 📏

### Mean Absolute Error (MAE) 😮
- **Interpretação:** Uma média absoluta dos erros entre as previsões e os valores reais. Quanto menor, melhor.
- **Exemplo de Uso:**
  ```python
  from sklearn.metrics import mean_absolute_error
  mae = mean_absolute_error(Y_test, previsoes)
  ```

### Mean Squared Error (MSE) 📉
- **Interpretação:** Uma média dos quadrados dos erros entre as previsões e os valores reais. Quanto menor, melhor.
- **Exemplo de Uso:**
  ```python
  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(Y_test, previsoes)
  ```

### R-squared (R²) 📈
- **Interpretação:** Coeficiente de determinação. Varia de 0 a 1, sendo 1 o melhor resultado. Indica a proporção da variância na variável dependente que é previsível a partir das variáveis independentes.
- **Exemplo de Uso:**
  ```python
  from sklearn.metrics import r2_score
  r2 = r2_score(Y_test, previsoes)
  ```
  Esses resultados sugerem que o modelo está se ajustando muito bem aos dados de treinamento. No entanto, é importante ter cuidado com a possibilidade de overfitting, especialmente se o modelo estiver muito complexo e ajustado demais aos dados de treinamento específicos.

  Além disso, ao considerar a qualidade do modelo, também é crucial avaliá-lo em um conjunto de teste independente para garantir que ele generalize bem para dados não vistos. Certifique-se de usar um conjunto de teste separado para avaliação adequada do desempenho do modelo.


## Visualização 3D Avançada 🌐📊

A visualização 3D interativa fornece insights sobre a distribuição espacial dos pontos de teste. Utilizamos a biblioteca Plotly para criar um ambiente exploratório imersivo.

```python
import plotly.graph_objects as go

# Criação do gráfico 3D
fig = go.Figure()

# Adição dos pontos reais e previstos
fig.add_trace(go.Scatter3d(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], z=Y_test, mode='markers', marker=dict(color='black', size=5), name='Real'))
fig.add_trace(go.Scatter3d(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], z=previsoes, mode='markers', marker=dict(color='red', size=5), name='Previsto'))

# Configuração do layout
fig.update_layout(scene=dict(xaxis_title='Primeira Variável Independente', yaxis_title='Segunda Variável Independente', zaxis_title='PRECIP. TOTAL (mm)'),
                  scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25)))

# Exibição do gráfico
fig.show()
```
- **Gráfico:**

 ![Gráfico de Dispersão - Regressão Linear](report/Linear%20Regression3D.png)

## Modelos de Regressão Avançada 🤖💡

### Regressão Linear
- **Descrição:** Modelo linear simples que assume uma relação linear entre variáveis independentes e dependentes.
- **Exemplo de Uso:**
  ```python
  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  model.fit(X_train, Y_train)
  previsoes = model.predict(X_test)
  ```
- **Avaliação:**
    - MAE: 63.93
    - MSE: 1567541.57
    - R-squared: 0.65
- **Gráfico:**

  ![Gráfico de Dispersão - Regressão Linear](report/Linear%20Regression.png)

### Árvore de Decisão
- **Descrição:** Modelo baseado em estruturas de árvore que divide o conjunto de dados em subconjuntos.
- **Exemplo de Uso:**
  ```python
  from sklearn.tree import DecisionTreeRegressor
  model = DecisionTreeRegressor()
  model.fit(X_train, Y_train)
  previsoes = model.predict(X_test)
  ```
- **Avaliação:**
    - MAE: 114.24
    - MSE: 258569.75
    - R-squared: 0.94
- **Gráfico:**

  ![Gráfico de Dispersão - Árvore de Decisão](report/Decision%20Tree.png)

### Random Forest
- **Descrição:** Ensemble de árvores de decisão, reduzindo overfitting e melhorando a precisão.
- **Exemplo de Uso:**
  ```python
  from sklearn.ensemble import RandomForestRegressor
  model = RandomForestRegressor()
  model.fit(X_train, Y_train)
  previsoes = model.predict(X_test)
  ```
- **Avaliação:**
    - MAE: 114.19
    - MSE: 137050.10
    - R-squared: 0.97
- **Gráfico:**

  ![Gráfico de Dispersão - Random Forest](report/Random%20Forest.png)

### Gradient Boosting
- **Descrição:** Ensemble de árvores treinadas sequencialmente, corrigindo erros dos modelos anteriores.
- **Exemplo de Uso:**
  ```python
  from sklearn.ensemble import GradientBoostingRegressor
  model = GradientBoostingRegressor()
  model.fit(X_train, Y_train)
  previsoes = model.predict(X_test)
  ```
- **Avaliação:**
    - MAE: 48.82
    - MSE: 51558.87
    - R-squared: 0.99
- **Gráfico:**

  ![Gráfico de Dispersão - Gradient Boosting](report/Gradient%20Boosting.png)

# Divisão do Conjunto de Dados para Treinamento e Teste 📂
```python
from sklearn.model_selection import train_test_split

# Dividindo o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
```

**Nota Importante:**
Os resultados apresentados podem variar devido ao parâmetro `test_size` nesta divisão de dados. O valor de `test_size` representa a proporção dos dados que serão utilizados como conjunto de teste, enquanto o restante é utilizado como conjunto de treinamento. 🔄

### Explicação:

- **`X` e `Y`:** Representam as características (features) e as variáveis de resposta (target), respectivamente, do conjunto de dados.

- **`test_size=0.1`:** Este parâmetro define a proporção do conjunto de dados que será utilizado como conjunto de teste. Neste caso, 10% dos dados serão reservados para teste, e 90% serão utilizados para treinamento. 🔍

- **`random_state=1`:** Este parâmetro é opcional, mas é usado para garantir reprodutibilidade. Ao definir um valor fixo para `random_state`, a divisão será sempre a mesma se o código for executado várias vezes com o mesmo conjunto de dados. Isso é útil para resultados consistentes e para evitar variações nos resultados durante o desenvolvimento e teste do modelo. 🚀

### Aviso:

É importante observar que a escolha do `test_size` influenciará diretamente na avaliação do desempenho do modelo. Um tamanho de teste muito pequeno pode levar a uma estimativa instável do desempenho, enquanto um tamanho muito grande pode resultar em uma diminuição na quantidade de dados disponíveis para treinamento.

Ao ajustar o valor de `test_size`, é recomendável considerar a quantidade total de dados disponíveis e o trade-off entre a precisão da avaliação do modelo e a quantidade de dados usada para treinamento. 🧐📊

## Escolha do Modelo e Considerações 🤔📈

### 1. Regressão Linear 🤖:

**Descrição:**
A Regressão Linear é um modelo simples que pressupõe uma relação linear entre as variáveis independentes e dependentes. Em outras palavras, ele tenta encontrar a melhor linha reta que se ajusta aos dados.

**Vantagens:**
- 📊 **Fácil interpretação:** Ideal quando a compreensão direta da relação entre variáveis é crucial.
- 💻 **Computacionalmente eficiente:** Rápido e eficaz em conjuntos de dados menores.
- 🔄 **Útil para relações lineares:** Funciona bem quando a relação entre variáveis é aproximadamente linear.

**Desvantagens:**
- 📈 **Limitado a relações lineares:** Pode não lidar bem com dados que têm relações não-lineares.
- 📉 **Sensibilidade a outliers:** Resultados podem ser afetados por valores extremos.

### 2. Árvore de Decisão 🌳:

**Descrição:**
A Árvore de Decisão é um modelo baseado em estruturas de árvore que divide o conjunto de dados em subconjuntos, tomando decisões em cada nó. Cada divisão é baseada em uma condição que maximiza a pureza dos subconjuntos resultantes.

**Vantagens:**
- 🌳 **Lida bem com relações não-lineares:** Capaz de capturar padrões complexos nos dados.
- 🚫 **Não requer normalização de variáveis:** Robusto a diferentes escalas de variáveis.
- 📊 **Fácil interpretação:** Possibilita visualização clara do processo de decisão.

**Desvantagens:**
- 🚀 **Tendência a overfitting:** Especialmente em árvores profundas, pode se ajustar demais aos dados de treinamento.
- 📈 **Sensibilidade a pequenas variações nos dados:** Pequenas mudanças nos dados podem levar a grandes alterações na árvore.

### 3. Random Forest 🌲:

**Descrição:**
Random Forest é um ensemble de árvores de decisão, onde várias árvores são treinadas e a previsão final é a média (regressão) ou a votação (classificação) das previsões individuais.

**Vantagens:**
- 🌲 **Reduz overfitting:** Ao combinar várias árvores, ajuda a generalizar melhor para novos dados.
- 🌐 **Lida bem com relações não-lineares:** Capaz de capturar padrões complexos e interações.
- 🛡️ **Robusto a outliers:** Menos suscetível a ser influenciado por valores extremos.

**Desvantagens:**
- 🔍 **Menos interpretável:** Dificuldade em entender o processo de decisão quando comparado a uma única árvore.
  
### 4. Gradient Boosting 📈:

**Descrição:**
Gradient Boosting é um ensemble de árvores que são treinadas sequencialmente, corrigindo os erros dos modelos anteriores.

**Vantagens:**
- 📈 **Alta precisão preditiva:** Pode fornecer resultados altamente precisos.
- 🌐 **Lida bem com relações não-lineares:** Capaz de capturar padrões complexos e interações.
- 🛡️ **Robusto a outliers:** Menos sensível a valores extremos.

**Desvantagens:**
- 🎚️ **Sensibilidade a hiperparâmetros:** Requer ajuste cuidadoso dos parâmetros.
- 🕰️ **Tempo de treinamento mais longo:** Pode levar mais tempo em comparação com métodos mais simples.

### Escolha do Modelo:

- **Regressão Linear:** Use quando a relação é linear e interpretabilidade é crucial.
  
- **Árvore de Decisão:** Use quando as relações são não-lineares e interpretação fácil é desejada.

- **Random Forest:** Use quando precisar de alta precisão preditiva e interpretabilidade não é crítica.

- **Gradient Boosting:** Use quando a precisão é crucial, e você está disposto a ajustar hiperparâmetros.

A escolha entre modelos depende da natureza do problema, da interpretabilidade desejada e da necessidade de precisão. Experimentar vários modelos e ajustar hiperparâmetros é recomendado para encontrar a melhor solução.

## Tamanho do Conjunto de Teste 🧐📉

Ao diminuir o conjunto de teste, há variação estatística aumentada, menor representatividade e maior risco de overfitting. Conjuntos de teste maiores proporcionam avaliações mais estáveis e confiáveis do desempenho do modelo.

Este relatório fornece uma visão abrangente da modelagem realizada, incorporando conceitos avançados, técnicas de visualização e análises detalhadas para impulsionar o entendimento e aprimoramento dos modelos. 🚀🔍
