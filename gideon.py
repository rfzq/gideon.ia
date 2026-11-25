import json
import re
import webbrowser
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELO = "qwen3.5:4b"

SYSTEM_PROMPT = """
Você é o Gideon, um assistente parceiro, direto e prático.
Ajude a resolver problemas, testar ideias e escrever código sem enrolação.
Explique passo a passo quando algo for complexo e mantenha um tom natural e próximo.
Responda sempre com clareza e foco no próximo passo útil.

Quando o usuário pedir uma ação (ex.: tocar música, abrir site/app, buscar algo),
escreva uma resposta curta e adicione NO FINAL da resposta exatamente neste formato:
[[CMD: {"acao": "tocar_musica" | "abrir_url" | "buscar_web", "query": "termo ou URL", "alvo": "youtube_music" | "youtube" | "spotify" | "navegador"}]]

Não use [[CMD: ...]] em conversas normais.
"""

historico = [
    {"role": "system", "content": SYSTEM_PROMPT.strip()}
]


def extrair_e_limpar_comando(texto: str):
    match = re.search(r"\[\[CMD:\s*(\{.*?\})\s*\]\]", texto, flags=re.DOTALL)
    if not match:
        return texto.strip(), None

    comando = None
    try:
        comando = json.loads(match.group(1))
    except json.JSONDecodeError:
        comando = None

    texto_limpo = (texto[:match.start()] + texto[match.end():]).strip()
    return texto_limpo, comando


def executar_comando_local(comando: dict):
    if not isinstance(comando, dict) or not comando.get("acao"):
        return

    acao = str(comando.get("acao")).lower()
    query = comando.get("query") or ""
    alvo = str(comando.get("alvo") or "").lower()

    if acao == "tocar_musica" and query:
        q = requests.utils.quote(query)
        if alvo == "spotify":
            url = f"https://open.spotify.com/search/{q}"
        elif alvo == "youtube":
            url = f"https://www.youtube.com/results?search_query={q}"
        else:
            url = f"https://music.youtube.com/search?q={q}"
        print(f"[CMD] Abrindo: {url}")
        webbrowser.open(url)
    elif acao == "abrir_url" and query.startswith(("http://", "https://")):
        print(f"[CMD] Abrindo: {query}")
        webbrowser.open(query)
    elif acao == "buscar_web" and query:
        q = requests.utils.quote(query)
        url = f"https://www.google.com/search?q={q}"
        print(f"[CMD] Buscando: {url}")
        webbrowser.open(url)


print("Gideon iniciado com qwen3.5:4b! Digite 'sair' para encerrar.\n")

while True:
    mensagem = input("Você: ").strip()
    if not mensagem:
        continue
    if mensagem.lower() in ("sair", "exit", "quit"):
        print("Gideon: Até mais!")
        break

    historico.append({"role": "user", "content": mensagem})

    resposta = requests.post(
        OLLAMA_URL,
        json={
            "model": MODELO,
            "messages": historico,
            "stream": False
        },
        timeout=120
    )
    resposta.raise_for_status()

    texto_ia = resposta.json()["message"]["content"]
    texto_limpo, comando = extrair_e_limpar_comando(texto_ia)
    historico.append({"role": "assistant", "content": texto_limpo})

    print(f"\nGideon: {texto_limpo}\n")
    if comando:
        executar_comando_local(comando)
