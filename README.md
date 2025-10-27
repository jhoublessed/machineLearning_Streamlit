# Projeto: App Previsor de Desempenho Acadêmico

Este projeto é um aplicativo web desenvolvido para prever o desempenho acadêmico de alunos, ajudando instituições de ensino a identificar estudantes em "zona de risco" antes que seja tarde demais.

## 🎯 O Problema

Instituições de ensino enfrentam um desafio comum: muitos alunos desistem ou são reprovados ao final do semestre, quando as possibilidades de intervenção pedagógica já são limitadas.

### A Dor

Coordenadores e professores, muitas vezes, não conseguem identificar facilmente quais alunos estão caminhando para a reprovação ou evasão antes das avaliações finais, perdendo a janela de oportunidade para agir.

## 💡 A Solução Proposta

Para resolver esse problema, construímos um **Aplicativo Web** utilizando **Streamlit**. A plataforma permite que um coordenador ou professor insira os dados relevantes de um aluno. Com base nessas informações, um modelo de **Inteligência Artificial (IA)** prevê a probabilidade desse aluno ser "Aprovado" ou "Reprovado".

## ✨ O "Pulo do Gato"

A eficiência do aplicativo está em sua arquitetura: ele não realiza o treinamento do modelo em tempo real. Em vez disso, o app consome um modelo de Machine Learning (ML) robusto, que já foi treinado com dados de um repositório online e está pronto para fazer predições instantâneas.

## 🚀 Resultado Final

O aplicativo será publicado e hospedado na nuvem (utilizando plataformas como Streamlit Cloud, Heroku, etc.), tornando-o acessível para que qualquer pessoa com o link possa utilizá-lo de forma prática e rápida.

## 🛠️ Tecnologias Utilizadas

* **Python**
* **Streamlit:** Para a construção da interface web.
* **Machine Learning:** Um modelo de ML (ex: Regressão Logística, Random Forest) treinado para classificação.
* **Pandas/Numpy:** (Provável) Para manipulação de dados.

## 🏁 Como Executar (Exemplo)

1.  Clone o repositório:
    ```bash
    git clone [URL-DO-SEU-REPOSITORIO]
    cd [NOME-DO-PROJETO]
    ```

2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3.  Execute o aplicativo Streamlit:
    ```bash
    streamlit run app.py
    ```
