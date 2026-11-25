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

Para "tocar_musica", coloque em "query" o nome da música e/ou artista (ex.: "Mac DeMarco Chamber Of Reflection").
Não use [[CMD: ...]] em conversas normais.
"""

historico = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
CMD_REGEX = re.compile(r"\[\[CMD:\s*(\{.*?\})\s*\]\]", flags=re.DOTALL)


def montar_mensagens():
    if len(historico) <= 1 + MAX_MENSAGENS_HISTORICO:
        return historico
    return [historico[0]] + historico[-MAX_MENSAGENS_HISTORICO:]


def limpar_cmd_do_historico(texto: str) -> str:
    return CMD_REGEX.sub("", texto).strip()


async def buscar_preview_audio(termo: str) -> str | None:
    termo_limpo = (termo or "").strip()
    if not termo_limpo:
        return None

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://itunes.apple.com/search",
                params={
                    "term": termo_limpo,
                    "media": "music",
                    "entity": "song",
                    "limit": 1,
                },
            )
            resp.raise_for_status()
            dados = resp.json()
            resultados = dados.get("results") or []
            if resultados:
                preview_url = resultados[0].get("previewUrl")
                if isinstance(preview_url, str) and preview_url.startswith(("http://", "https://")):
                    return preview_url
    except Exception as erro:
        print(f"Aviso ao buscar prévia de áudio: {erro}")

    return None


async def enriquecer_comandos_no_texto(texto: str) -> str:
    matches = list(CMD_REGEX.finditer(texto))
    if not matches:
        return texto

    novo_texto = texto
    for match in reversed(matches):
        bloco_json_bruto = match.group(1)
        try:
            cmd = json.loads(bloco_json_bruto)
        except Exception:
            continue

        if not isinstance(cmd, dict):
            continue

        acao = str(cmd.get("acao") or "").lower()
        url_existente = cmd.get("url") or cmd.get("audio_url") or ""
        query = str(cmd.get("query") or "").strip()

        if acao == "tocar_musica" and not url_existente and query:
            preview_url = await buscar_preview_audio(query)
            if preview_url:
                cmd["url"] = preview_url
                cmd_substituido = f"[[CMD: {json.dumps(cmd, ensure_ascii=False)}]]"
                inicio, fim = match.span()
                novo_texto = novo_texto[:inicio] + cmd_substituido + novo_texto[fim:]

    return novo_texto


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
        buffer_enviado = 0
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
                            texto_acumulado = "".join(resposta_completa)
                            # Segura o [[CMD: ...]] até o fim para poder injetar a "url" de prévia
                            idx_cmd = texto_acumulado.find("[[CMD:")
                            limite_seguro = len(texto_acumulado) if idx_cmd == -1 else idx_cmd
                            if limite_seguro > buffer_enviado:
                                yield texto_acumulado[buffer_enviado:limite_seguro]
                                buffer_enviado = limite_seguro

            texto_final = "".join(resposta_completa).strip()
            texto_enriquecido = await enriquecer_comandos_no_texto(texto_final)
            if len(texto_enriquecido) > buffer_enviado:
                yield texto_enriquecido[buffer_enviado:]

            if texto_enriquecido:
                historico.append({"role": "assistant", "content": limpar_cmd_do_historico(texto_enriquecido)})
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
