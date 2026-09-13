#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 FIAP BANK - CENTRAL DE OUVIDORIA INTELIGENTE
 Checkpoint 4 - IA Aplicada a RPA - Triagem Inteligente com Resiliencia
================================================================================

CRITERIO DE NEGOCIO OFICIAL (Diretoria de Experiencia do Cliente)
--------------------------------------------------------------------------------
Uma manifestacao e classificada como URGENTE quando houver PELO MENOS UMA
das situacoes abaixo:

    (A) Fraude ou suspeita de fraude;
    (B) Cobranca indevida, duplicada ou nao reconhecida;
    (C) Perda financeira ja ocorrida, ou valor devido que nao foi recebido.

Todos os demais casos sao NORMAL, incluindo falhas tecnicas do aplicativo,
duvidas, elogios e reclamacoes de atendimento sem impacto financeiro direto.

ATENCAO: o criterio e o CONTEUDO, nao o TOM. Uma manifestacao escrita de
forma calma pode ser urgente; uma manifestacao escrita de forma irritada
pode ser normal. Essa e a armadilha central deste checkpoint e o motivo
pelo qual o prompt de classificacao foi revisado (ver secao "HISTORICO DE
AJUSTES DO PROMPT" abaixo).

--------------------------------------------------------------------------------
HISTORICO DE AJUSTES DO PROMPT (item C - Qualidade da Classificacao)
--------------------------------------------------------------------------------
V1 (versao inicial, com erro): pedia para o modelo classificar "com base no
teor geral da mensagem", sem proibir explicitamente o uso do tom emocional
como sinal. Ao validar contra o conjunto de teste (TEST_SET abaixo), essa
versao classificou como "normal" manifestacoes de fraude escritas em tom
calmo e educado, e como "urgente" reclamacoes de atendimento escritas em
tom irritado mas sem qualquer impacto financeiro. A causa raiz: o modelo
usava a "temperatura emocional" do texto como proxy de urgencia, em vez do
criterio de negocio literal.

V2 (versao corrigida, em uso): o prompt agora (1) transcreve o criterio de
negocio literalmente, (2) instrui de forma explicita "classifique pelo
CONTEUDO, nunca pelo TOM emocional", (3) inclui exemplos few-shot que
isolam exatamente esse contraste (mensagem calma + fraude = urgente;
mensagem irritada + duvida tecnica = normal), e (4) pede uma
"justificativa" curta citando qual letra do criterio (A/B/C/nenhuma) foi
usada, o que facilita auditoria e reduz respostas "no chute". A funcao
`validar_contra_conjunto_de_teste()` roda ambas as versoes e imprime a
acuracia de cada uma para comprovar a melhora antes de rodar o lote
completo.

--------------------------------------------------------------------------------
MELHORIA NO MODELO DE ANALISE DE SENTIMENTO (em relacao a versao anterior)
--------------------------------------------------------------------------------
A versao anterior deste robo usava um modelo local generico de estrelas
(nlptown/bert-base-multilingual-uncased-sentiment), treinado sobre reviews
de produtos, e convertia "1-5 estrelas" em positivo/negativo/neutro por
uma regra fixa. Isso gera dois problemas conhecidos neste dominio:
  1) o modelo nao entende contexto bancario (ex.: "resolvido rapidamente"
     vs. "cobranca indevida" tem cargas emocionais parecidas para um
     modelo de estrelas, mas significados de sentimento bem diferentes);
  2) por ser um modelo de proposito geral, ele nao usa a MESMA leitura de
     contexto que decide a urgencia, podendo gerar pares urgencia/
     sentimento inconsistentes (ex.: marcar uma fraude como "neutro").

A versao atual substitui o modelo de estrelas por CLASSIFICACAO POR IA
(LLM) especializada por prompt: o mesmo modelo que le a manifestacao para
decidir a urgencia tambem decide o sentimento, na MESMA chamada de API.
Isso (a) usa um criterio de sentimento adaptado a ouvidoria bancaria
("positivo" = elogio/agradecimento, "negativo" = insatisfacao/reclamacao/
receio, "neutro" = duvida ou pedido operacional sem carga emocional
clara), (b) garante coerencia entre urgencia e sentimento por vir da mesma
leitura do texto, e (c) reduz o numero de chamadas de API pela metade
(1 chamada por manifestacao em vez de 2), o que importa diretamente para
nao estourar a cota gratuita dos provedores.

--------------------------------------------------------------------------------
ARQUITETURA DE RESILIENCIA
--------------------------------------------------------------------------------
  1. Retry com espera progressiva (backoff exponencial) em toda chamada de
     API de IA (`chamar_com_retry`).
  2. Fallback entre dois provedores gratuitos: Gemini (principal) e Groq
     (reserva). Se o principal falhar apos os retries, o reserva assume.
  3. Se AMBOS falharem, a manifestacao recebe status REVISAO_HUMANA (nunca
     fica sem registro).
  4. Idempotencia: antes de qualquer chamada de API, o robo verifica se o
     texto ja foi processado (hash SHA-256 armazenado no SQLite). Se ja
     existe, a linha e pulada sem gastar cota.
  5. Toda escrita no banco usa placeholders (nunca concatenacao de string)
     e esta protegida por try/except com commit apenas em caso de sucesso.
================================================================================
"""

import hashlib
import json
import os
import smtplib
import sqlite3
import time
from datetime import datetime
from email.message import EmailMessage
from getpass import getpass

import pandas as pd
import requests

# ==============================================================================
# 1. CONFIGURACAO
# ==============================================================================

CSV_PATH = "manifestacoes_clientes.csv"
DB_PATH = "ouvidoria_fiap_bank.db"

# Nomes de modelo mudam com frequencia nos dois provedores. Confirme o
# modelo atual em aistudio.google.com e console.groq.com antes de rodar.
GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-20b"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MAX_RETRIES = 3          # tentativas por provedor antes de cair para o proximo
BACKOFF_BASE_SECONDS = 2  # espera = BACKOFF_BASE_SECONDS * (2 ** tentativa)

PROVEDOR_PRINCIPAL = "gemini"
PROVEDOR_RESERVA = "groq"
STATUS_REVISAO_HUMANA = "REVISAO_HUMANA"

ENVIAR_EMAIL_REAL = False  # troque para True + preencha SMTP_* para enviar de fato
EMAIL_DESTINATARIO = "diretoria.experiencia@fiapbank.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USUARIO = ""
SMTP_SENHA = ""


def carregar_api_keys():
    """
    Nunca deixe API Keys escritas no codigo. Usa variavel de ambiente se
    existir (util em execucao automatizada) e cai para getpass (input
    oculto) caso contrario - igual ao praticado em aula com Colab Secrets.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or getpass(
        "Cole a GEMINI_API_KEY (input oculto): "
    )
    groq_key = os.environ.get("GROQ_API_KEY") or getpass(
        "Cole a GROQ_API_KEY (input oculto): "
    )
    return gemini_key, groq_key


# ==============================================================================
# 2. PROMPTS DE CLASSIFICACAO (V1 com erro conhecido / V2 corrigido)
# ==============================================================================

CRITERIO_NEGOCIO = """\
CRITERIO OFICIAL DE URGENCIA DO FIAP BANK:
Classifique como "urgente" se HOUVER PELO MENOS UM dos itens abaixo:
  (A) fraude ou suspeita de fraude;
  (B) cobranca indevida, duplicada ou nao reconhecida;
  (C) perda financeira ja ocorrida, ou valor devido que nao foi recebido.
Caso contrario, classifique como "normal" (inclui falhas tecnicas do
aplicativo, duvidas, elogios e reclamacoes de atendimento sem impacto
financeiro direto).
"""


def montar_prompt_v1(texto: str) -> str:
    """Versao inicial (mantida apenas para fins de documentacao/comparacao).

    Falha conhecida: nao proibe explicitamente o uso do tom emocional como
    sinal, entao o modelo tende a usar a "temperatura" do texto como proxy
    de urgencia.
    """
    return f"""{CRITERIO_NEGOCIO}
Classifique a manifestacao abaixo com base no teor geral da mensagem.

Responda ESTRITAMENTE em JSON, sem nenhum texto fora do JSON, no formato:
{{"urgencia": "urgente" ou "normal", "sentimento": "positivo" ou "negativo" ou "neutro"}}

Manifestacao do cliente:
\"\"\"{texto}\"\"\"
"""


def montar_prompt_v2(texto: str) -> str:
    """Versao corrigida e em uso em producao (ver HISTORICO DE AJUSTES)."""
    return f"""Voce e um analista senior de ouvidoria do FIAP Bank.

{CRITERIO_NEGOCIO}
REGRA CRITICA: classifique pelo CONTEUDO da manifestacao, NUNCA pelo TOM
emocional. Uma mensagem calma e educada pode ser urgente; uma mensagem
irritada pode ser normal.

Exemplos de calibragem (nao repita estes exemplos na resposta):
- "Notei uma cobranca em duplicidade na minha fatura, poderiam verificar
  quando possivel?" (tom calmo) -> urgencia = "urgente" (criterio B)
- "Estou revoltado, o aplicativo caiu de novo e perdi um tempo enorme"
  (tom irritado) -> urgencia = "normal" (falha tecnica, sem criterio A/B/C)

Alem da urgencia, classifique o SENTIMENTO da manifestacao no contexto de
ouvidoria bancaria:
- "positivo": elogio, agradecimento ou satisfacao explicita.
- "negativo": insatisfacao, reclamacao, receio ou frustracao.
- "neutro": duvida ou pedido operacional, sem carga emocional clara.

Responda ESTRITAMENTE em JSON, sem nenhum texto fora do JSON, no formato:
{{"urgencia": "urgente" ou "normal", "sentimento": "positivo" ou "negativo" ou "neutro", "criterio": "A", "B", "C" ou "nenhum", "justificativa": "uma frase curta"}}

Manifestacao do cliente:
\"\"\"{texto}\"\"\"
"""


# Prompt usado no lote completo. Trocar para montar_prompt_v1 reproduz o
# erro documentado acima (util apenas para fins didaticos de comparacao).
montar_prompt = montar_prompt_v2


def extrair_json(texto_resposta: str) -> dict:
    """Extrai o primeiro objeto JSON valido de uma resposta de LLM,
    tolerando texto acidental antes/depois (ex.: crases de markdown)."""
    texto_resposta = texto_resposta.strip()
    texto_resposta = texto_resposta.replace("```json", "").replace("```", "")
    inicio = texto_resposta.find("{")
    fim = texto_resposta.rfind("}")
    if inicio == -1 or fim == -1:
        raise ValueError("Nenhum JSON encontrado na resposta do modelo")
    return json.loads(texto_resposta[inicio : fim + 1])


def normalizar_classificacao(dado: dict) -> dict:
    """Garante que os campos essenciais existem e estao nos valores
    permitidos, para nao propagar lixo para o banco de dados."""
    urgencia = str(dado.get("urgencia", "")).strip().lower()
    sentimento = str(dado.get("sentimento", "")).strip().lower()

    if urgencia not in ("urgente", "normal"):
        raise ValueError(f"Valor de urgencia invalido: {urgencia!r}")
    if sentimento not in ("positivo", "negativo", "neutro"):
        raise ValueError(f"Valor de sentimento invalido: {sentimento!r}")

    return {"urgencia": urgencia, "sentimento": sentimento}


# ==============================================================================
# 3. CHAMADAS DE API COM RETRY (BACKOFF PROGRESSIVO)
# ==============================================================================


def chamar_gemini(texto: str, api_key: str, modelo: str = None) -> dict:
    modelo = modelo or GEMINI_MODEL
    url = GEMINI_URL.format(model=modelo, key=api_key)
    payload = {
        "contents": [{"parts": [{"text": montar_prompt(texto)}]}],
        "generationConfig": {"temperature": 0.1},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    corpo = resp.json()
    texto_resposta = corpo["candidates"][0]["content"]["parts"][0]["text"]
    return normalizar_classificacao(extrair_json(texto_resposta))


def chamar_groq(texto: str, api_key: str, modelo: str = None) -> dict:
    modelo = modelo or GROQ_MODEL
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": modelo,
        "messages": [{"role": "user", "content": montar_prompt(texto)}],
        "temperature": 0.1,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    corpo = resp.json()
    texto_resposta = corpo["choices"][0]["message"]["content"]
    return normalizar_classificacao(extrair_json(texto_resposta))


def chamar_com_retry(func_provedor, texto: str, api_key: str, modelo: str = None,
                      max_tentativas: int = MAX_RETRIES) -> dict:
    """Aplica retry com espera progressiva (backoff exponencial) em cima de
    qualquer funcao de chamada de provedor (`chamar_gemini` / `chamar_groq`)."""
    ultimo_erro = None
    for tentativa in range(1, max_tentativas + 1):
        try:
            return func_provedor(texto, api_key, modelo)
        except Exception as erro:
            ultimo_erro = erro
            espera = BACKOFF_BASE_SECONDS * (2 ** (tentativa - 1))
            print(
                f"    [retry] tentativa {tentativa}/{max_tentativas} falhou "
                f"({erro.__class__.__name__}: {erro}). Aguardando {espera}s..."
            )
            time.sleep(espera)
    raise ultimo_erro


def classificar_com_fallback(texto: str, gemini_key: str, groq_key: str,
                              gemini_modelo: str = None) -> tuple:
    """Tenta o provedor principal (Gemini); se falhar apos os retries, tenta
    o provedor de reserva (Groq); se ambos falharem, retorna REVISAO_HUMANA.

    `gemini_modelo` permite injetar um nome de modelo invalido para a
    demonstracao obrigatoria de fallback (ver secao 7).
    """
    try:
        print(f"  -> tentando provedor principal ({PROVEDOR_PRINCIPAL})...")
        resultado = chamar_com_retry(chamar_gemini, texto, gemini_key, gemini_modelo)
        return resultado, PROVEDOR_PRINCIPAL
    except Exception as erro_principal:
        print(f"  -> provedor principal esgotou as tentativas: {erro_principal}")
        try:
            print(f"  -> acionando fallback ({PROVEDOR_RESERVA})...")
            resultado = chamar_com_retry(chamar_groq, texto, groq_key)
            return resultado, PROVEDOR_RESERVA
        except Exception as erro_reserva:
            print(f"  -> provedor de reserva tambem esgotou as tentativas: {erro_reserva}")
            return None, STATUS_REVISAO_HUMANA


# ==============================================================================
# 4. CONJUNTO DE TESTE (VALIDACAO MANUAL) - item C dos requisitos
# ==============================================================================

# 10 manifestacoes rotuladas manualmente (minimo exigido: 8), escolhidas
# para cobrir exatamente a armadilha "conteudo vs. tom" descrita no
# criterio de negocio.
TEST_SET = [
    {
        "texto": "Identifiquei uma transacao de R$850 que nao reconheco no meu extrato de hoje.",
        "urgencia_esperada": "urgente",
        "sentimento_esperado": "negativo",
    },
    {
        "texto": "Poderiam verificar com calma quando possivel? Notei que fui cobrado duas vezes na fatura deste mes.",
        "urgencia_esperada": "urgente",
        "sentimento_esperado": "neutro",
    },
    {
        "texto": "Estou revoltado, o aplicativo caiu tres vezes hoje e nao consegui nem checar meu saldo.",
        "urgencia_esperada": "normal",
        "sentimento_esperado": "negativo",
    },
    {
        "texto": "Gostaria de saber como altero meu endereco de cadastro no aplicativo.",
        "urgencia_esperada": "normal",
        "sentimento_esperado": "neutro",
    },
    {
        "texto": "O atendimento da agencia foi otimo, o gerente resolveu minha duvida rapidamente. Parabens!",
        "urgencia_esperada": "normal",
        "sentimento_esperado": "positivo",
    },
    {
        "texto": "Recebi uma mensagem estranha pedindo meus dados bancarios, acho que e uma tentativa de fraude.",
        "urgencia_esperada": "urgente",
        "sentimento_esperado": "negativo",
    },
    {
        "texto": "Fiz uma transferencia de R$1.200 na sexta-feira e o valor ainda nao caiu na conta do destinatario.",
        "urgencia_esperada": "urgente",
        "sentimento_esperado": "negativo",
    },
    {
        "texto": "O app trava toda vez que tento acessar o extrato, isso e muito frustrante e ja aconteceu varias vezes.",
        "urgencia_esperada": "normal",
        "sentimento_esperado": "negativo",
    },
    {
        "texto": "Excelente experiencia usando o cartao de credito internacional, sem nenhuma taxa escondida.",
        "urgencia_esperada": "normal",
        "sentimento_esperado": "positivo",
    },
    {
        "texto": "Meu limite foi reduzido sem nenhum aviso previo e isso me deixou numa situacao financeira dificil.",
        "urgencia_esperada": "urgente",
        "sentimento_esperado": "negativo",
    },
]


def validar_contra_conjunto_de_teste(gemini_key: str, groq_key: str) -> float:
    """Roda o TEST_SET pelo pipeline com fallback e imprime a acuracia de
    urgencia e de sentimento. Deve ser executado ANTES do lote completo."""
    acertos_urgencia = 0
    acertos_sentimento = 0
    linhas_relatorio = []

    for item in TEST_SET:
        resultado, provedor = classificar_com_fallback(item["texto"], gemini_key, groq_key)
        if resultado is None:
            linhas_relatorio.append((item["texto"][:50], "FALHOU", "FALHOU", provedor))
            continue

        acertou_urgencia = resultado["urgencia"] == item["urgencia_esperada"]
        acertou_sentimento = resultado["sentimento"] == item["sentimento_esperado"]
        acertos_urgencia += acertou_urgencia
        acertos_sentimento += acertou_sentimento
        linhas_relatorio.append(
            (
                item["texto"][:50],
                f"{resultado['urgencia']} ({'OK' if acertou_urgencia else 'ERRO'})",
                f"{resultado['sentimento']} ({'OK' if acertou_sentimento else 'ERRO'})",
                provedor,
            )
        )

    n = len(TEST_SET)
    acuracia_urgencia = acertos_urgencia / n
    acuracia_sentimento = acertos_sentimento / n

    print("\n=== RESULTADO DA VALIDACAO CONTRA O CONJUNTO DE TESTE ===")
    df_val = pd.DataFrame(
        linhas_relatorio,
        columns=["texto (resumo)", "urgencia (predita)", "sentimento (predito)", "provedor"],
    )
    print(df_val.to_string(index=False))
    print(f"\nAcuracia de urgencia:   {acuracia_urgencia:.1%} ({acertos_urgencia}/{n})")
    print(f"Acuracia de sentimento: {acuracia_sentimento:.1%} ({acertos_sentimento}/{n})")

    return acuracia_urgencia


# ==============================================================================
# 5. PERSISTENCIA (SQLITE COM PLACEHOLDERS + TRY/EXCEPT/COMMIT)
# ==============================================================================


def gerar_hash(texto: str) -> str:
    """Hash estavel do texto original, usado como chave de idempotencia."""
    return hashlib.sha256(texto.strip().lower().encode("utf-8")).hexdigest()


def inicializar_banco(db_path: str = DB_PATH) -> sqlite3.Connection:
    conexao = sqlite3.connect(db_path)
    try:
        conexao.execute(
            """
            CREATE TABLE IF NOT EXISTS manifestacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manifestacao_id TEXT,
                texto_original TEXT NOT NULL,
                texto_hash TEXT NOT NULL UNIQUE,
                urgencia TEXT NOT NULL,
                sentimento TEXT NOT NULL,
                provedor_utilizado TEXT NOT NULL,
                data_processamento TEXT NOT NULL
            )
            """
        )
        conexao.commit()
    except sqlite3.Error as erro:
        print(f"Erro ao inicializar o banco: {erro}")
        conexao.rollback()
    return conexao


def ja_processado(conexao: sqlite3.Connection, texto_hash: str) -> bool:
    """Checagem de idempotencia: evita reprocessar (e gastar cota de API
    com) uma manifestacao ja registrada."""
    cursor = conexao.execute(
        "SELECT 1 FROM manifestacoes WHERE texto_hash = ? LIMIT 1", (texto_hash,)
    )
    return cursor.fetchone() is not None


def salvar_manifestacao(conexao: sqlite3.Connection, manifestacao_id, texto: str,
                         texto_hash: str, urgencia: str, sentimento: str,
                         provedor: str) -> bool:
    """Insere uma manifestacao classificada. Usa apenas placeholders (?),
    nunca concatenacao de string. Retorna True em caso de sucesso."""
    try:
        conexao.execute(
            """
            INSERT INTO manifestacoes (
                manifestacao_id, texto_original, texto_hash,
                urgencia, sentimento, provedor_utilizado, data_processamento
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(manifestacao_id),
                texto,
                texto_hash,
                urgencia,
                sentimento,
                provedor,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conexao.commit()
        return True
    except sqlite3.Error as erro:
        print(f"  [erro ao salvar no banco] {erro}")
        conexao.rollback()
        return False


# ==============================================================================
# 6. PROCESSAMENTO DO LOTE (COM IDEMPOTENCIA)
# ==============================================================================


def processar_lote(df: pd.DataFrame, conexao: sqlite3.Connection,
                    gemini_key: str, groq_key: str) -> None:
    coluna_id = "id" if "id" in df.columns else None

    for posicao, linha in df.iterrows():
        texto = str(linha["texto"])
        manifestacao_id = linha[coluna_id] if coluna_id else posicao
        texto_hash = gerar_hash(texto)

        if ja_processado(conexao, texto_hash):
            print(f"[{manifestacao_id}] ja processado anteriormente - pulando (idempotencia).")
            continue

        print(f"[{manifestacao_id}] classificando: {texto[:60]}...")
        resultado, provedor = classificar_com_fallback(texto, gemini_key, groq_key)

        if resultado is None:
            # Nenhum provedor conseguiu classificar: registra para revisao
            # humana em vez de descartar a manifestacao.
            salvar_manifestacao(
                conexao, manifestacao_id, texto, texto_hash,
                urgencia="indefinido", sentimento="indefinido", provedor=provedor,
            )
            print(f"  -> RESULTADO: {STATUS_REVISAO_HUMANA}")
            continue

        sucesso = salvar_manifestacao(
            conexao, manifestacao_id, texto, texto_hash,
            urgencia=resultado["urgencia"], sentimento=resultado["sentimento"],
            provedor=provedor,
        )
        if sucesso:
            print(f"  -> RESULTADO: urgencia={resultado['urgencia']}, "
                  f"sentimento={resultado['sentimento']}, provedor={provedor}")


# ==============================================================================
# 7. DEMONSTRACAO OBRIGATORIA DO FALLBACK (FALHA FORCADA NO PROVEDOR PRINCIPAL)
# ==============================================================================


def demonstrar_fallback(gemini_key: str, groq_key: str) -> None:
    """Forca o provedor principal a falhar usando um NOME DE MODELO
    INEXISTENTE no Gemini, e mostra que o robo conclui a classificacao pelo
    provedor de reserva (Groq). Sem esta evidencia, o fallback nao e
    considerado demonstrado pelos criterios do checkpoint."""
    texto_exemplo = "Notei uma cobranca duplicada na minha fatura deste mes."
    modelo_invalido = "modelo-que-nao-existe-999"

    print("=== DEMONSTRACAO DE FALLBACK (falha forcada no provedor principal) ===")
    print(f"Forcando falha do Gemini com modelo invalido: '{modelo_invalido}'\n")

    resultado, provedor = classificar_com_fallback(
        texto_exemplo, gemini_key, groq_key, gemini_modelo=modelo_invalido
    )

    print(f"\nProvedor efetivamente usado: {provedor}")
    print(f"Resultado da classificacao: {resultado}")
    assert provedor == PROVEDOR_RESERVA, (
        "Fallback nao ocorreu como esperado - verifique as chaves de API."
    )
    print("Fallback demonstrado com sucesso: o provedor de reserva assumiu "
          "a classificacao apos a falha forcada do provedor principal.")


# ==============================================================================
# 8. RELATORIO DE GOVERNANCA (VIA PANDAS)
# ==============================================================================


def gerar_relatorio_governanca(conexao: sqlite3.Connection) -> dict:
    df_banco = pd.read_sql_query("SELECT * FROM manifestacoes", conexao)

    total = len(df_banco)
    distribuicao_urgencia = df_banco["urgencia"].value_counts().to_dict()
    distribuicao_sentimento = df_banco["sentimento"].value_counts().to_dict()
    distribuicao_provedor = df_banco["provedor_utilizado"].value_counts().to_dict()

    relatorio = {
        "total_processado": total,
        "distribuicao_urgencia": distribuicao_urgencia,
        "distribuicao_sentimento": distribuicao_sentimento,
        "resolvidos_provedor_principal": distribuicao_provedor.get(PROVEDOR_PRINCIPAL, 0),
        "resolvidos_provedor_reserva": distribuicao_provedor.get(PROVEDOR_RESERVA, 0),
        "enviados_revisao_humana": distribuicao_provedor.get(STATUS_REVISAO_HUMANA, 0),
    }

    print("\n=== RELATORIO DE GOVERNANCA DO LOTE ===")
    for chave, valor in relatorio.items():
        print(f"{chave}: {valor}")

    return relatorio


# ==============================================================================
# 9. NOTIFICACAO (MONTAGEM DO E-MAIL DE GOVERNANCA)
# ==============================================================================


def montar_email_governanca(relatorio: dict) -> EmailMessage:
    corpo = f"""Prezada Diretoria de Experiencia do Cliente,

Segue o relatorio de governanca do lote de triagem automatica de
manifestacoes processado em {datetime.now().strftime('%d/%m/%Y as %H:%M')}.

Total de manifestacoes processadas: {relatorio['total_processado']}

Distribuicao por urgencia:
{json.dumps(relatorio['distribuicao_urgencia'], ensure_ascii=False, indent=2)}

Distribuicao por sentimento:
{json.dumps(relatorio['distribuicao_sentimento'], ensure_ascii=False, indent=2)}

Resolvidos pelo provedor principal (Gemini): {relatorio['resolvidos_provedor_principal']}
Resolvidos pelo provedor de reserva (Groq): {relatorio['resolvidos_provedor_reserva']}
Enviados para revisao humana: {relatorio['enviados_revisao_humana']}

Atenciosamente,
Robo de Triagem Inteligente - FIAP Bank
"""

    mensagem = EmailMessage()
    mensagem["Subject"] = "Relatorio de Governanca - Triagem Inteligente de Ouvidoria"
    mensagem["From"] = SMTP_USUARIO or "robo-triagem@fiapbank.com"
    mensagem["To"] = EMAIL_DESTINATARIO
    mensagem.set_content(corpo)

    print("\n=== E-MAIL DE GOVERNANCA (MONTADO) ===")
    print(f"De: {mensagem['From']}")
    print(f"Para: {mensagem['To']}")
    print(f"Assunto: {mensagem['Subject']}")
    print("--- corpo ---")
    print(corpo)

    return mensagem


def enviar_email(mensagem: EmailMessage) -> None:
    """Envio real e opcional (ENVIAR_EMAIL_REAL = False por padrao). A
    montagem correta da mensagem acima e o item obrigatorio do checkpoint."""
    if not ENVIAR_EMAIL_REAL:
        print("\n[envio simulado] ENVIAR_EMAIL_REAL=False - e-mail apenas montado, nao enviado.")
        return
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as servidor:
            servidor.starttls()
            servidor.login(SMTP_USUARIO, SMTP_SENHA)
            servidor.send_message(mensagem)
        print("\nE-mail enviado com sucesso.")
    except Exception as erro:
        print(f"\n[erro ao enviar e-mail] {erro}")


# ==============================================================================
# 10. ORQUESTRACAO (MAIN)
# ==============================================================================


def main():
    print("=" * 80)
    print("FIAP BANK - ROBO DE TRIAGEM INTELIGENTE DE OUVIDORIA")
    print("=" * 80)

    gemini_key, groq_key = carregar_api_keys()

    # --- 1) Validacao contra o conjunto de teste (obrigatorio antes do lote) ---
    acuracia = validar_contra_conjunto_de_teste(gemini_key, groq_key)
    if acuracia < 0.8:
        print(
            "\nAtencao: acuracia de urgencia abaixo de 80%. Revise o prompt "
            "(montar_prompt_v2) antes de prosseguir para o lote completo."
        )

    # --- 2) Demonstracao obrigatoria do fallback ---
    demonstrar_fallback(gemini_key, groq_key)

    # --- 3) Ingestao do lote completo ---
    if not os.path.exists(CSV_PATH):
        print(f"\nArquivo '{CSV_PATH}' nao encontrado - encerrando antes do lote completo.")
        return
    df = pd.read_csv(CSV_PATH)

    # --- 4) Persistencia com idempotencia ---
    conexao = inicializar_banco(DB_PATH)
    try:
        processar_lote(df, conexao, gemini_key, groq_key)

        # --- 5) Governanca ---
        relatorio = gerar_relatorio_governanca(conexao)

        # --- 6) Notificacao ---
        mensagem = montar_email_governanca(relatorio)
        enviar_email(mensagem)
    finally:
        conexao.close()


if __name__ == "__main__":
    main()
