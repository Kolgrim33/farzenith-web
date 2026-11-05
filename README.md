# FarZenith web

A chat page for [FarZenith](https://huggingface.co/Kolgrim/farzenith-small-v1), the Zimbabwean
Astronomical Society's astronomy model. Ask about orbits, satellites, stars, black holes, the
expanding universe or Zimbabwe's dark skies. FarZenith picks an experiment, a physics engine
computes every number, and FarZenith explains the result. Follow-ups that change one number
("What about at 2000 km?") keep the conversation's context.

The model runs in NumPy (`zenith.serve.lite`), so the app needs no PyTorch and fits a 512 MB server.

## Run locally

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860
```

Or with Docker:

```bash
docker build -t farzenith-web .
docker run -p 7860:7860 farzenith-web
```

## Deploy on Render

`render.yaml` describes a free Docker web service. In Render: **New > Blueprint**, paste this
repository's URL, and apply. Render sets `PORT`; the app listens on it.

## API

`POST /gradio_api/call/ask` with `{"data": ["your question"]}` returns an event id; read the answer
from `GET /gradio_api/call/ask/<event id>`. Python: `gradio_client.Client(url).predict(q, api_name="/ask")`.

`hf-space/README.md` is the Hugging Face Space card, kept for when the app is hosted there.

Built by Selby Mashiki for the Zimbabwean Astronomical Society. Apache 2.0.
