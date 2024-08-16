# Projeto de Análise de Dados Meteorológicos com Chatbot Inteligente

Este repositório contém um projeto desenvolvido como parte da disciplina CPS769 - Introdução à Inteligência Artificial e Aprendizado Generativo. O projeto é uma aplicação de análise de dados meteorológicos que utiliza um chatbot inteligente para responder a perguntas sobre temperaturas máximas, mínimas e médias com base em dados históricos.

## Descrição do Projeto

O projeto envolve a criação de um chatbot que pode consultar e processar dados meteorológicos armazenados em um arquivo CSV. O chatbot utiliza o modelo GPT-4o-mini da OpenAI e é integrado com ferramentas que permitem a análise dos dados meteorológicos, tais como a busca pela temperatura máxima, mínima e média para anos e meses específicos.

### Funcionalidades

- **Análise de Dados Meteorológicos**: O código lê dados de um arquivo CSV e processa informações sobre temperatura e outros parâmetros meteorológicos.
- **Chatbot Inteligente**: Utiliza o modelo GPT-4o-mini para interagir com o usuário e responder a consultas sobre dados meteorológicos.
- **Ferramentas de Consulta**: O chatbot pode buscar a temperatura máxima, mínima e média para anos e meses específicos e comparar com outros anos se desejado.

## Dependências

O projeto utiliza as seguintes bibliotecas e ferramentas:

- `pandas`: Para manipulação e análise de dados.
- `langchain_core`: Para integração com o modelo GPT-4o-mini e definição de classes de ferramentas.
- `langchain_openai`: Para interação com o modelo GPT-4o-mini da OpenAI.
- `python-dotenv`: Para carregamento de variáveis de ambiente.

## Estrutura do Projeto

- **`data/weather_2000.csv`**: Arquivo CSV contendo os dados meteorológicos.
- **`script.py`**: Script principal que carrega dados, processa informações e executa o chatbot.

## Como Usar

1. **Instale as dependências**:

   ```bash
   pip install requirements.txt
   ```

2. **Configure as variáveis de ambiente**: Certifique-se de que o arquivo `.env` está configurado com as credenciais necessárias para acessar o modelo da OpenAI.

3. **Execute o script**:

   ```bash
   python script.py
   ```

4. **Interaja com o chatbot**: Digite perguntas relacionadas a temperaturas máximas, mínimas e médias. Para sair do chatbot, digite `exit()`.

## Exemplos de Consultas

- "Qual é a temperatura máxima registrada em janeiro de 2020?"
- "Qual foi a temperatura mínima em julho de 2019 comparada com 2018?"
- "Qual é a média de temperatura para março de 2021?"
