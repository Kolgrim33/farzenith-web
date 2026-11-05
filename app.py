"""FarZenith: chat about astronomy; the model picks an experiment and a physics engine computes it."""
import json
import os
import sys
import tempfile

import gradio as gr
from huggingface_hub import snapshot_download

MODEL_DIR = snapshot_download("Kolgrim/farzenith-small-v1")
sys.path.insert(0, os.environ.get("ZENITH_CODE", os.path.join(MODEL_DIR, "code")))  # ZENITH_CODE: local testing
os.environ.setdefault("ZENITH_DARKSKY", MODEL_DIR)

from zenith.engines.base import Result  # noqa: E402
from zenith.engines.plot import save_plot  # noqa: E402
from zenith.engines.registry import simulate  # noqa: E402
from zenith.format import Turn  # noqa: E402
from zenith.serve.lite import ZenithLite  # noqa: E402  (NumPy back end: fits a 512 MB server)

zen = ZenithLite.load(MODEL_DIR)
MAX_TURNS = 6  # earlier turns the model sees; Zenith.fit_history trims further to fit the context

# The FarZenith logo (star trails around the pole), turned into the "thinking" indicator.
SPINNER = """
<div class="fz-thinking" role="status">
  <svg viewBox="0 0 200 200" width="44" height="44" aria-hidden="true">
    <g fill="none" stroke="currentColor" stroke-width="10" stroke-linecap="round">
      <path class="fz-outer" d="M100 30 A70 70 0 1 1 30 100"/>
      <path class="fz-inner" d="M138 100 A38 38 0 1 1 100 62"/>
    </g>
    <circle cx="100" cy="100" r="12" fill="#E3B752"/>
  </svg>
  <span>Running the experiment. This takes a few seconds.</span>
</div>
"""


def _trails():
    """Static star trails around the south celestial pole, as a long exposure from Zimbabwe shows."""
    import random
    rnd = random.Random(18)  # fixed, so the sky is the same on every load
    arcs = []
    for _ in range(110):
        r = 18 + 760 * rnd.random() ** 1.4  # more trails close to the pole
        a0 = rnd.uniform(0, 360)
        sweep = rnd.uniform(16, 30)  # same exposure for every star: similar angular length
        op = rnd.uniform(0.12, 0.5) * (1.0 if r < 320 else 0.7)
        w = rnd.choice((0.9, 1, 1, 1.3, 1.8))
        arcs.append(f'<circle r="{r:.0f}" fill="none" stroke="#F2ECDD" stroke-opacity="{op:.2f}" '
                    f'stroke-width="{w}" stroke-linecap="round" pathLength="360" '
                    f'stroke-dasharray="{sweep:.1f} 360" transform="rotate({a0:.0f})"/>')
    return ('<svg class="fz-trails" viewBox="-900 -560 1100 640" preserveAspectRatio="xMaxYMax slice" '
            'aria-hidden="true">' + "".join(arcs) + '<circle r="4" fill="#E3B752"/></svg>')


HEADER = f"""
<header class="fz-head">
  {_trails()}
  <div class="fz-head-text">
    <h1>FarZenith</h1>
    <p class="fz-lede">Ask about orbits, satellites, stars, black holes, the expanding universe or
    Zimbabwe's dark skies. FarZenith turns your question into an experiment, a physics engine computes
    every number, and FarZenith explains what came out.</p>
    <p class="fz-by">From the Zimbabwean Astronomical Society</p>
  </div>
</header>
"""

CSS = """
footer{display:none !important}
.gradio-container{max-width:1180px !important;margin:0 auto !important;padding:0 16px 40px !important}
.fz-head-wrap{padding:0 !important;border:0 !important;background:none !important}
.fz-head{position:relative;overflow:hidden;border-radius:4px;margin:16px -12px 4px;
  background:radial-gradient(120% 140% at 100% 100%, #16243A 0%, #0E1828 55%, #0A111D 100%);
  border:1px solid #1E2C42;min-height:240px;display:flex;align-items:flex-end}
.fz-trails{position:absolute;inset:0;width:100%;height:100%}
.fz-head-text{position:relative;z-index:1;padding:36px 32px 28px;max-width:40rem}
.fz-head h1{font-family:'Spectral',Georgia,serif;font-weight:500;font-size:clamp(2.6rem,6vw,4.2rem);
  line-height:1;letter-spacing:-0.01em;margin:0 0 14px;color:#F2ECDD}
.fz-lede{font-family:'Spectral',Georgia,serif;font-size:1.12rem;line-height:1.6;color:#D9D3C4;margin:0 0 14px}
.fz-by{font-size:.9rem;color:#8E99AB;margin:0}
.fz-chat .message{font-family:'Spectral',Georgia,serif;font-size:1.06rem;line-height:1.65}
.fz-chat .message.user, .fz-chat .user{font-family:'IBM Plex Sans',sans-serif;font-size:.98rem}
.fz-thinking{display:flex;align-items:center;gap:14px;padding:6px 2px;color:#F2ECDD;font-size:.95rem}
.fz-thinking span{color:#8E99AB}
.fz-thinking svg{flex:none}
.fz-outer,.fz-inner{transform-origin:100px 100px}
.fz-outer{animation:fz-spin 2.6s linear infinite}
.fz-inner{animation:fz-spin 1.6s linear infinite reverse}
@keyframes fz-spin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion: reduce){.fz-outer,.fz-inner{animation-duration:9s}}
.fz-plot img{background:#F2ECDD;border-radius:3px}
.fz-input button{background:#E3B752 !important;color:#0E1828 !important;font-weight:600}
.fz-input button:hover{background:#E9C66F !important}
.fz-new{align-self:flex-start;width:auto !important;flex:none !important;min-width:0 !important}
.fz-note{color:#8E99AB;font-size:.88rem;max-width:44rem}
.fz-note a{color:#E3B752}
@media (max-width:700px){.fz-chat{height:340px !important}.fz-trails{opacity:.55}.fz-head{min-height:200px}.fz-head-text{padding:28px 20px 22px}.fz-lede{font-size:1.02rem}}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,500;1,400&display=swap">')
FORCE_DARK = "() => { document.body.classList.add('dark'); }"


def _turns(past):
    """Rebuild model history from the (question, spec, answer) triples kept in the session."""
    turns = []
    for q, spec, answer in past[-MAX_TURNS:]:
        try:
            turns.append(Turn(q, spec, simulate(spec), answer))
        except Exception:  # a spec the engines reject is left out of the history
            pass
    return turns


def _run(question, past):
    r = zen.ask(question.strip()[:1000], _turns(past))
    plot = None
    if r.result and r.result.get("series"):
        plot = os.path.join(tempfile.mkdtemp(), "plot.png")
        save_plot(Result(**r.result), plot)
    answer = r.answer or (f"Sorry, I couldn't answer that: {r.error}" if r.error else "Sorry, I have no answer.")
    if r.attribution:
        answer += f"\n\n_{r.attribution}_"
    if r.spec and r.answer:
        past = past + [(question, r.spec, r.answer)]
    return answer, (json.dumps(r.spec, indent=2) if r.spec else ""), plot, past


def start(question, messages):
    """Show the question and the logo spinner straight away."""
    if not question or not question.strip():
        return gr.skip(), messages, gr.skip(), gr.update(visible=False)
    messages = messages + [{"role": "user", "content": question.strip()}]
    return "", messages, question.strip(), gr.update(visible=True)


def answer(pending, messages, past):
    if not pending:
        return messages, gr.skip(), gr.skip(), past, gr.update(visible=False), ""
    text, spec, plot, past = _run(pending, past)
    messages = messages + [{"role": "assistant", "content": text}]
    return messages, spec, gr.update(value=plot, visible=plot is not None), past, gr.update(visible=False), ""


def ask(question):
    """One-shot API: answer, experiment spec (JSON) and plot."""
    text, spec, plot, _ = _run(question, [])
    return text, spec, plot


EXAMPLES = ["How long does the ISS take to orbit Earth?", "Where around Zimbabwe can I get the darkest skies?",
            "How big is a black hole of ten Suns?", "What colour is a 10,000 K star?",
            "How far away is a galaxy at redshift 2?"]

GOLD = gr.themes.Color(c50="#FBF4E2", c100="#F6E7BF", c200="#EFD593", c300="#E9C66F", c400="#E3B752",
                       c500="#D2A23A", c600="#B5872C", c700="#8E6923", c800="#6A4F1C", c900="#4A3814",
                       c950="#2C210B")
NIGHT = gr.themes.Color(c50="#F2ECDD", c100="#D9D3C4", c200="#B8BFCB", c300="#8E99AB", c400="#6A7689",
                        c500="#4A566B", c600="#2E3B52", c700="#1E2C42", c800="#142034", c900="#0E1828",
                        c950="#0A111D")
THEME = gr.themes.Base(primary_hue=GOLD, neutral_hue=NIGHT, radius_size=gr.themes.sizes.radius_sm,
                       font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"]).set(
    body_background_fill_dark="#0E1828", body_text_color_dark="#F2ECDD",
    block_background_fill_dark="#0A111D", block_border_color_dark="#1E2C42", block_label_text_color_dark="#8E99AB",
    block_title_text_color_dark="#B8BFCB", input_background_fill_dark="#0E1828", input_border_color_dark="#2E3B52",
    input_border_color_focus_dark="#E3B752", button_primary_background_fill_dark="#E3B752",
    button_primary_background_fill_hover_dark="#E9C66F", button_primary_text_color_dark="#0E1828",
    button_secondary_background_fill_dark="#142034", button_secondary_text_color_dark="#D9D3C4",
    color_accent_soft_dark="#142034", link_text_color_dark="#E3B752")

with gr.Blocks(title="FarZenith") as app:
    gr.HTML(HEADER, elem_classes="fz-head-wrap", padding=False)
    past = gr.State([])
    pending = gr.State("")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            bot = gr.Chatbot(show_label=False, height=440, elem_classes="fz-chat",
                             placeholder="Ask a question below. You can follow up, for example "
                                         "\u201cWhat about the Moon?\u201d")
            thinking = gr.HTML(SPINNER, visible=False)
            q = gr.Textbox(show_label=False, placeholder="How long does the ISS take to orbit Earth?",
                           submit_btn="Ask", max_length=1000, elem_classes="fz-input")
            gr.Examples(EXAMPLES, q, label="Try one of these")
            clear = gr.Button("Start a new conversation", size="sm", variant="secondary", elem_classes="fz-new")
        with gr.Column(scale=2):
            plot = gr.Image(label="Plot of the latest experiment", show_label=False, elem_classes="fz-plot",
                            interactive=False, visible=False)
            with gr.Accordion("Experiment details", open=False):
                spec = gr.Code(show_label=False, language="json")
    gr.Markdown("FarZenith small v1 is a 50-million-parameter model, built from scratch and still in beta. It "
                "sometimes picks the wrong experiment, so check the plot and the experiment details. "
                "[About the model](https://huggingface.co/Kolgrim/farzenith-small-v1)", elem_classes="fz-note")
    q.submit(start, [q, bot], [q, bot, pending, thinking], show_progress="hidden", queue=False).then(
        answer, [pending, bot, past], [bot, spec, plot, past, thinking, pending], show_progress="hidden",
        api_name=False)
    clear.click(lambda: ([], [], "", gr.update(value=None, visible=False)), None, [bot, past, spec, plot], api_name=False)

    with gr.Row(visible=False):
        api_q = gr.Textbox()
        api_a, api_spec, api_plot = gr.Markdown(), gr.Code(), gr.Image()
        gr.Button().click(ask, api_q, [api_a, api_spec, api_plot], api_name="ask")

app.queue(default_concurrency_limit=1, max_size=20)
app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=THEME, css=CSS, js=FORCE_DARK, head=FONTS)
