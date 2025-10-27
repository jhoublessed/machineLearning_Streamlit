import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing,pipeline,linear_model, metrics


def carregar_dados(caminho_arquivo = "historicoAcademico.csv"):
    try:
        if os.path.exists(caminho_arquivo):

            df = pd.read_csv(caminho_arquivo, encoding="latin1",sep=',')

            print("o arquivo foi carregado com sucesso")

            return df
        else:
            print("o arquivo não foi encontrado dentro da pasta")

            return None
    except Exception as e:
        print("erro inesperado ao caregar o arquivo")

        return None
    
#--- chamar a função para armazenar o resultado ---#

dados = carregar_dados()


# --------- ETAPA 02 : PREPARAÇÃO E DIVISÃO DOS DADOS ----------- #
# definição de X (features) e Y (target)

if dados is not None:
    print(f"\ntotal de registros carregados: {len(dados)}")
    print("iniciando pipeline de treinamento")

    TARGET_COLUMN = "Status_Final"

    try:
        X = dados.drop(TARGET_COLUMN,axis=1)
        y = dados[TARGET_COLUMN]

        print(f"Features (X) definidas: {list(X.columns)}")
        print(f"Features (y) definidas: {TARGET_COLUMN}")
    except KeyError:
        print(f"\n ------ Erro critico ------ ")
        print(f"A coluna {TARGET_COLUMN} não foi encontrada no CSV")
        print(f"Colunas disponiveis: {list(dados.columns)}")
        print(f"Por favor, ajuste a variável 'TARGET_COLUMN' e tente novamente")
        exit()


    print("\n ----- Dividindo dados em treino e teste... -----")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )

    print(f"Dados de treino: {len(X_train)} Dados de teste: {len(X_test)}")

    print("\n ----- Criando a pupeline de ML... ----- ")
    pipeline_model = pipeline.Pipeline([
        ('scaler',preprocessing.StandardScaler()),
        ('model',linear_model.LogisticRegression(random_state=42))
    ])

    print("\n ---- Treinamento do modelo... ----")

    pipeline_model.fit(X_train,y_train)

    print("modelo treinando. avaliando com os dados de teste")
    y_pred = pipeline_model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred)

    print("\n ----- Relatorio de avaliação geral -----")
    print(f"\n acuracia geral: {accuracy * 100:.2f}%")
    print("\n relatorio de classificação detalhado")
    print(report)


    model_filename = 'modelo_previsao_desempenho.joblib'

    print(f"\nSalvando o pipeline treinado e..{model_filename}")
    joblib.dump(pipeline_model,model_filename)

    print("processo concluido")
    print(f"o arquivo'{model_filename} esta para ser utilizado")

else:
    print("o pipeline nao pode continuar pois os dados nao forma carregados")
    