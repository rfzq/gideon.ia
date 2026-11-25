import json
import os
import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx

app = FastAPI(title="Gideon")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
MODELO = os.getenv("GIDEON_MODEL", "openai/gpt-4o-mini")
MAX_MENSAGENS_HISTORICO = 6

SYSTEM_PROMPT = """
És o Gideon, um assistente direto e prático.
Foste criado e configurado pelo Rafa.
Responde de forma curta e objetiva, salvo quando pedirem código ou detalhe.

Quando o usuário pedir uma ação (ex.: tocar música, abrir site/app, buscar algo),
escreva uma resposta curta e adicione NO FINAL da resposta exatamente neste formato:
[[CMD: {"acao": "tocar_musica" | "abrir_url" | "buscar_web", "query": "termo ou URL", "alvo": "youtube_music" | "youtube" | "spotify" | "navegador", "url": "URL_DIRETA_OPCIONAL"}]]

Use "url" apenas se tiver uma URL direta de áudio/mídia conhecida; caso contrário, omita "url".
Não use [[CMD: ...]] em conversas normais.
"""

historico = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]


def montar_mensagens():
    if len(historico) <= 1 + MAX_MENSAGENS_HISTORICO:
        return historico
    return [historico[0]] + historico[-MAX_MENSAGENS_HISTORICO:]


def limpar_cmd_do_historico(texto: str) -> str:
    return re.sub(r"\[\[CMD:\s*\{.*?\}\s*\]\]", "", texto, flags=re.DOTALL).strip()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "modelo": MODELO, "model_name": MODELO},
    )


@app.post("/chat")
@app.post("/api/chat")
async def chat(request: Request):
    dados = await request.json()
    mensagem = (dados.get("mensagem") or dados.get("message") or "").strip()
    if not mensagem:
        return {"erro": "Mensagem vazia"}

    if not LLM_API_KEY:
        return {"erro": "LLM_API_KEY não configurada"}

    historico.append({"role": "user", "content": mensagem})
    mensagens_envio = montar_mensagens()

    async def gerar_stream():
        resposta_completa = []
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload_req = {
            "model": MODELO,
            "messages": mensagens_envio,
            "stream": True,
            "max_tokens": 250,
            "temperature": 0.5,
        }
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream(
                    "POST",
                    LLM_API_URL,
                    headers=headers,
                    json=payload_req,
                ) as resp:
                    resp.raise_for_status()
                    async for linha in resp.aiter_lines():
                        if not linha:
                            continue
                        if linha.startswith("data: "):
                            linha = linha[6:].strip()
                        if linha == "[DONE]":
                            break
                        payload = json.loads(linha)
                        delta = payload.get("choices", [{}])[0].get("delta", {})
                        parte = delta.get("content") or ""
                        if parte:
                            resposta_completa.append(parte)
                            yield parte
            texto_final = "".join(resposta_completa).strip()
            if texto_final:
                historico.append({"role": "assistant", "content": limpar_cmd_do_historico(texto_final)})
        except Exception as erro:
            yield f"\n[Erro: {erro}]"

    return StreamingResponse(
        gerar_stream(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},
    )


@app.post("/reset")
@app.post("/api/reset")
async def reset():
    global historico
    historico = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
    return {"status": "ok"}
