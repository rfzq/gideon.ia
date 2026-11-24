import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELO = "qwen3.5:4b"

SYSTEM_PROMPT = """
Você é o Gideon, um assistente parceiro, direto e prático.
Ajude a resolver problemas, testar ideias e escrever código sem enrolação.
Explique passo a passo quando algo for complexo e mantenha um tom natural e próximo.
Responda sempre com clareza e foco no próximo passo útil.
"""

historico = [
    {"role": "system", "content": SYSTEM_PROMPT.strip()}
]

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
    historico.append({"role": "assistant", "content": texto_ia})

    print(f"\nGideon: {texto_ia}\n")