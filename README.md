WhisperVeil Tarot 🔮
====================

Minimal Streamlit app that gives supportive tarot readings using the OpenAI Python SDK.

Quick Start
-----------

1) Create a virtualenv (optional but recommended)

   - `python3 -m venv .venv && source .venv/bin/activate`

2) Install dependencies

   - `pip install -r requirements.txt`

3) Provide your API key

   - Copy env file: `cp .env.example .env` and edit values
   - Or set shell vars: `export OPENAI_API_KEY=sk-...` and optionally `export OPENAI_MODEL=gpt-4o-mini`
   - (Optional) Pick a different image model (defaults to `dall-e-3`): `export OPENAI_IMAGE_MODEL=dall-e-3`
   - (Optional) Change image size (must be supported by your model): `export OPENAI_IMAGE_SIZE=1024x1024`

4) Run the app

   - `streamlit run app.py`

Features (MVP)
--------------

- 1-card and 3-card spreads (upright/reversed)
- Optional DALL-E 3 tarot art for each drawn card, with six visual themes
- Chat UI with history, redraw, and art regeneration buttons
- Clear reflection/entertainment disclaimer

Notes
-----

- All card logic runs locally; the chat model interprets only the drawn cards.
- Image generation only triggers when you toggle "Show card art," and prompts stay consistent per theme so every spread shares a cohesive style.
- For a more “agent-like” setup, you can expose a Python tool (e.g., `draw_tarot_cards`) and let the model call it via tool/function calling, while still drawing cards deterministically on the Python side.

Project Layout
--------------

- `app.py` — Streamlit app and tarot logic
- `requirements.txt` — Dependencies
