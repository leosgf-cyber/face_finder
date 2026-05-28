# Face Finder — Contexto do Projeto

> Ferramenta de reconhecimento facial em vídeos pra editores de casamento (e outros casos).
> Workflow: cadastra pessoas via fotos/pastas/vídeos → sobe vídeos → app identifica quem aparece em quais frames.

## Stack

| Camada | Tech |
|---|---|
| Backend | Python + Flask (modo debug com hot-reload) |
| Reconhecimento facial | `dlib` (HOG detector + ResNet encoder) — **sem** `face_recognition` (tinha bugs de packaging) |
| Processamento de vídeo | `ffmpeg` (proxy 1080p, frames configuráveis por FPS) |
| Frontend | HTML + CSS + JS vanilla (dark UI minimalista) |
| Async | `threading` (vai migrar pra Celery se virar SaaS) |

## Como rodar

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./dev.sh    # auto-pull a cada 30s + hot-reload Flask + browser auto-reload
```

Acessa http://localhost:8080

**Pré-requisitos do sistema:** `ffmpeg`, `cmake`, build tools (pra compilar dlib).
**Python:** 3.12 (não funciona em 3.14 — `face_recognition_models` quebra).

## Estrutura

```
face_finder/
├── extractor.py          # ffmpeg wrapper: extrai frames com timestamps configuráveis
├── scanner.py            # dlib direto: detecção, encoding, clustering, scan de frames
├── main.py               # CLI standalone (sem UI)
├── dev.sh                # Auto-pull + Flask + browser reload
├── requirements.txt
└── web/
    ├── app.py            # Flask API
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/app.js
```

## Funcionalidades implementadas

### Cadastro de pessoas (3 modos, todos com clustering automático)
1. **Upload manual** — sobe fotos + digita nome
2. **Apontar pasta** — abre Finder, seleciona pasta, sistema agrupa rostos similares automaticamente
3. **Vídeo de referência** — sobe clip, extrai frames, agrupa rostos

Auto-match: se um rosto escaneado já está cadastrado, o nome é preenchido automaticamente com tag "Já cadastrado(a)".

### UI de clustering
- Paginação (8 por página)
- Botão X pra pular rosto (não cadastra)
- **Seleção múltipla** + barra flutuante pra combinar clusters da mesma pessoa em um nome (ex: foto de óculos + sem óculos)
- Confirmação ao mudar nome já preenchido
- Slider global de sensibilidade (0.35-0.70) — controla agressividade do agrupamento

### Vídeos
- Upload de múltiplos vídeos
- Apontar pasta inteira via Finder
- Gestão (listar/remover)

### Processamento
- FPS configurável
- Timestamps início/fim (ex: 01:33 → 22:21)
- Tolerância de match configurável
- Roda async em background com polling
- Progresso em tempo real: matches parciais durante processamento
- Output: lista por pessoa com timestamps, thumbnails recortados dos rostos detectados, JSON pra download

### Otimizações
- Proxy 1080p antes de processar (ffmpeg)
- Resize pra 1200px de largura antes do dlib
- Cache de encodings das referências (pickle, valida por mtime)
- Skip de frames similares (compara histograma reduzido)
- Limpeza automática de frames após processamento
- Botão manual "Limpar temporários"
- Limite upload 2GB

## Arquitetura de clustering

Função `_cluster_faces(faces, tolerance=0.55, merge_tolerance=0.45)`:

1. **Best-match clustering**: cada rosto comparado com a MÉDIA dos encodings de cada cluster (não só o primeiro). Atribui ao cluster mais próximo se distância ≤ tolerance.
2. **Merge pass**: depois do primeiro round, funde clusters cuja distância média entre eles seja ≤ merge_tolerance (mais conservador, evita over-merge).
3. **Ordenação**: clusters ordenados por número de aparições (mais frequente primeiro).

A merge_tolerance é deliberadamente mais estrita que a tolerance inicial — evita juntar pessoas diferentes que têm encodings parecidos.

## Workflow de dev (importante)

`dev.sh` faz auto-pull do git a cada 30s. Endpoint `/api/version` retorna mtime dos arquivos static/templates; o frontend pollea e auto-reload. Flask em debug recarrega `.py` automaticamente.

**Resultado:** push no remoto → 30s depois Mac puxa → Flask reinicia → navegador recarrega. Zero interação.

## Ideias na fila (não implementadas)

- **Smart FPS** — detectar cortes de cena (scenedetect) e extrair só nos cuts, em vez de FPS fixo. Esforço alto, valor alto.
- **Processamento paralelo** — multiprocessing pra usar todos os cores no scan
- **Resultados progressivos por frame** (não só por vídeo)
- **Confiança da identificação no cluster** — mostrar quão certo o sistema está de cada agrupamento
- **Treinamento de tolerância** — sugerir tolerância baseado nos rostos do scan atual
- **Histórico de processamentos** — não perder jobs ao reiniciar
- **Multi-usuário / auth** — se virar SaaS
- **Fila com Celery + Redis** — substituir threading
- **GPU** — usar CUDA pro dlib (se disponível)

## Decisões importantes

- **Por que dlib direto e não `face_recognition`**: a lib `face_recognition` tem bugs de packaging em Python moderno (`face_recognition_models` falha com `pkg_resources`). Como ela é só um wrapper fino do dlib, fomos direto pro dlib.
- **Por que threading e não Celery**: MVP local pra uso pessoal. Migrar pra Celery quando virar SaaS.
- **Por que Flask e não FastAPI**: simplicidade pra MVP, tudo num projeto só. FastAPI faria sentido se separássemos backend/frontend.
- **Modelos baixados sob demanda**: na primeira execução, `scanner.py` baixa os modelos do dlib (~100MB) automaticamente do GitHub do davisking/dlib-models.

## Pontos de atenção

- O cache de encodings (`references/.encodings_cache.pkl`) precisa ser invalidado quando referências são adicionadas/removidas — feito via mtime check.
- Pastas pesadas: avisar usuário com `confirm()` se >200 imagens.
- AirPlay do macOS ocupa porta 5000 — por isso usamos 8080.
- Pra repos com muitas pessoas (>50 referências), o auto-match na hora do scan-folder pode ficar lento — considerar batch ou cache.
