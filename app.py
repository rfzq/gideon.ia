import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import httpx

app = FastAPI(title="Gideon")
templates = Jinja2Templates(directory="templates")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODELO = os.getenv("GIDEON_MODEL", "qwen3.5:4b")
MAX_MENSAGENS_HISTORICO = 6

SYSTEM_PROMPT = """
És o Gideon, um assistente direto e prático.
Foste criado e configurado pelo Rafa como um projeto local com FastAPI e Ollama.
Responde de forma curta e objetiva, salvo quando pedirem código ou detalhe.
"""

historico = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]


def montar_mensagens():
    if len(historico) <= 1 + MAX_MENSAGENS_HISTORICO:
        return historico
    return [historico[0]] + historico[-MAX_MENSAGENS_HISTORICO:]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "modelo": MODELO},
    )


@app.post("/chat")
async def chat(request: Request):
    dados = await request.json()
    mensagem = (dados.get("mensagem") or "").strip()
    if not mensagem:
        return {"erro": "Mensagem vazia"}

    historico.append({"role": "user", "content": mensagem})
    mensagens_envio = montar_mensagens()

    async def gerar_stream():
        resposta_completa = []
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream(
                    "POST",
                    OLLAMA_URL,
                    json={
                        "model": MODELO,
                        "messages": mensagens_envio,
                        "stream": True,
                        "think": False,
                        "keep_alive": "30m",
                        "options": {
                            "num_ctx": 1024,
                            "num_predict": 120,
                            "temperature": 0.5,
                        },
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for linha in resp.aiter_lines():
                        if not linha:
                            continue
                        payload = json.loads(linha)
                        parte = payload.get("message", {}).get("content", "")
                        if parte:
                            resposta_completa.append(parte)
                            yield parte
            texto_final = "".join(resposta_completa).strip()
            if texto_final:
                historico.append({"role": "assistant", "content": texto_final})
        except Exception as erro:
            yield f"\n[Erro: {erro}]"

    return StreamingResponse(
        gerar_stream(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},
    )

