import base64
import os
import random

import concurrent.futures

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables and configure the page
load_dotenv()
st.set_page_config(page_title="WhisperVeil Tarot", page_icon="🔮")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")
IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None


# Simple Major Arcana (upright/reversed in short phrases)
MAJOR_ARCANA = [
    {"name": "The Fool", "upright": "beginnings, innocence, leap of faith", "reversed": "hesitation, recklessness, holding back"},
    {"name": "The Magician", "upright": "manifestation, resourcefulness, willpower", "reversed": "manipulation, scattered energy"},
    {"name": "The High Priestess", "upright": "intuition, inner voice, mystery", "reversed": "secrets, doubt, disconnected self"},
    {"name": "The Empress", "upright": "nurturing, creativity, abundance", "reversed": "blocks, dependence, creative lull"},
    {"name": "The Emperor", "upright": "structure, leadership, stability", "reversed": "rigidity, control issues"},
    {"name": "The Hierophant", "upright": "tradition, learning, guidance", "reversed": "rebellion, question norms"},
    {"name": "The Lovers", "upright": "union, values, choice", "reversed": "misalignment, indecision"},
    {"name": "The Chariot", "upright": "drive, willpower, victory", "reversed": "scattered aim, stalled momentum"},
    {"name": "Strength", "upright": "compassion, courage, steady power", "reversed": "self-doubt, impatience"},
    {"name": "The Hermit", "upright": "reflection, wisdom, solitude", "reversed": "isolation, avoidance"},
    {"name": "Wheel of Fortune", "upright": "change, cycles, luck", "reversed": "resistance, feeling stuck"},
    {"name": "Justice", "upright": "truth, fairness, cause/effect", "reversed": "bias, imbalance"},
    {"name": "The Hanged Man", "upright": "surrender, new perspective", "reversed": "stalling, fear of letting go"},
    {"name": "Death", "upright": "endings, transformation, renewal", "reversed": "clinging, delayed change"},
    {"name": "Temperance", "upright": "balance, moderation, synthesis", "reversed": "excess, discord"},
    {"name": "The Devil", "upright": "attachments, shadow, temptation", "reversed": "release, reclaiming power"},
    {"name": "The Tower", "upright": "sudden change, revelation", "reversed": "fear of upheaval"},
    {"name": "The Star", "upright": "hope, healing, guidance", "reversed": "discouragement, doubt"},
    {"name": "The Moon", "upright": "intuition, uncertainty, dreams", "reversed": "clarity emerging, anxiety lifts"},
    {"name": "The Sun", "upright": "joy, vitality, success", "reversed": "temporary clouds, burnout risk"},
    {"name": "Judgement", "upright": "awakening, reckoning, call", "reversed": "self-critique, hesitation"},
    {"name": "The World", "upright": "completion, wholeness, integration", "reversed": "loose ends, near-finish"},
]

SPREADS = {
    "1-Card Daily": ["Focus"],
    "3-Card Past/Present/Future": ["Past", "Present", "Future"],
}

IMAGE_THEMES = {
    "Mystic Watercolor": {
        "prompt": "dreamy watercolor tarot illustration with luminous indigo, violet, and gold washes, soft brushstrokes, and a glowing celestial border",
    },
    "Celestial Midnight": {
        "prompt": "starlit midnight sky tarot card, deep navy and cobalt gradients, silver constellations, luminous moonlight, elegant filigree border",
    },
    "Art Deco Radiance": {
        "prompt": "luxurious art deco tarot design, symmetrical geometry, metallic gold accents, jewel-tone palette, sleek stylized figures, ornate frame",
    },
    "Forest Folk Magic": {
        "prompt": "whimsical forest folk tarot illustration, painterly textures, mossy greens, warm firefly glow, woodland spirits, carved wood border",
    },
    "Cyberpunk Neon": {
        "prompt": "futuristic cyberpunk tarot card, neon magenta and teal lighting, holographic glyphs, chrome details, digital circuitry border",
    },
    "Vintage Collage": {
        "prompt": "surreal vintage collage tarot artwork, aged paper textures, muted pastel palette, layered botanical motifs, delicate hand-inked border",
    },
}

SYSTEM_PROMPT = """You are a kind, grounded tarot reader.
- Use only the provided drawn cards and their positions.
- Offer supportive, actionable insights. No determinism or fatalism.
- Keep responses to ~200–300 words.
- Include a gentle disclaimer: this is for reflection/entertainment, not medical/legal/financial advice.
- If the question is vague, briefly clarify then interpret.
"""


def draw_cards(spread_key: str):
    positions = SPREADS[spread_key]
    deck = list(MAJOR_ARCANA)
    random.shuffle(deck)
    chosen = []
    for pos in positions:
        card = deck.pop()
        orientation = "upright" if random.random() >= 0.5 else "reversed"
        chosen.append(
            {
                "position": pos,
                "name": card["name"],
                "orientation": orientation,
                "meaning": card[orientation],
            }
        )
    return chosen


def cards_text(cards):
    return "\n".join(
        f"{c['position']}: {c['name']} ({c['orientation']}) — {c['meaning']}" for c in cards
    )


def card_signature(cards):
    return tuple(f"{c['position']}::{c['name']}::{c['orientation']}" for c in cards)


def generate_card_images(cards, theme_key):
    theme = IMAGE_THEMES[theme_key]

    def render_card(card):
        prompt = (
            f"{theme['prompt']} tarot card illustration of {card['name']} shown in a {card['orientation']} orientation. "
            f"Highlight symbolic elements of {card['meaning']}. Consistent ornate border, centered figure, rich texture, no extra text beyond the card title."
        )
        try:
            result = client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                size=theme.get("size", IMAGE_SIZE),
            )
            data = result.data[0]
            image_b64 = getattr(data, "b64_json", None)
            if image_b64:
                return base64.b64decode(image_b64), None
            image_url = getattr(data, "url", None)
            if image_url:
                return image_url, None
            return None, "Image generation returned no content."
        except Exception as exc:
            return None, str(exc)

    images = [None] * len(cards)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cards) or 1) as executor:
        future_map = {executor.submit(render_card, card): idx for idx, card in enumerate(cards)}
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            image_bytes, error_msg = future.result()
            images[idx] = image_bytes
            if error_msg:
                errors.append(error_msg)

    return images, errors


def ensure_card_images():
    cards = st.session_state.cards
    theme = st.session_state.theme
    if not cards:
        st.session_state.images = []
        st.session_state.image_error = ""
        return

    meta = st.session_state.get("image_meta") or {}
    signature = card_signature(cards)
    current_images = st.session_state.get("images", [])

    if (
        meta.get("theme") == theme
        and meta.get("signature") == signature
        and len(current_images) == len(cards)
        and all(img is not None for img in current_images)
    ):
        return

    if not client:
        st.session_state.images = []
        st.session_state.image_error = "missing_client"
        st.session_state.image_meta = {}
        return

    with st.spinner("Conjuring tarot artwork..."):
        images, errors = generate_card_images(cards, theme)

    st.session_state.images = images
    st.session_state.image_meta = {"theme": theme, "signature": signature}
    st.session_state.image_error = "; ".join(errors) if errors else ""


def generate_answer(prompt: str) -> str:
    if not client:
        return "I need an API key to interpret the cards. Please set OPENAI_API_KEY and try again."
    context = (
        f"Drawn cards:\n{cards_text(st.session_state.cards)}\n\n"
        "Interpret using these positions only."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context},
        *st.session_state.messages,
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=600,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"There was an error generating the reading: {e}"


# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "spread" not in st.session_state:
    st.session_state.spread = "3-Card Past/Present/Future"
if "cards" not in st.session_state:
    st.session_state.cards = draw_cards(st.session_state.spread)
if "theme" not in st.session_state:
    st.session_state.theme = list(IMAGE_THEMES.keys())[0]
if "images" not in st.session_state:
    st.session_state.images = []
if "image_meta" not in st.session_state:
    st.session_state.image_meta = {"theme": None, "signature": None}
if "image_error" not in st.session_state:
    st.session_state.image_error = ""
if "show_art" not in st.session_state:
    st.session_state.show_art = False


# UI
st.title("🔮 WhisperVeil Tarot")
st.caption("For reflection and entertainment only.")

with st.sidebar:
    spread_options = list(SPREADS.keys())
    spread_index = spread_options.index(st.session_state.spread)
    selected_spread = st.selectbox("Choose a spread", spread_options, index=spread_index)
    if selected_spread != st.session_state.spread:
        st.session_state.spread = selected_spread
        st.session_state.cards = draw_cards(selected_spread)
        st.session_state.images = []
        st.session_state.image_meta = {}
        st.session_state.messages = []
        st.session_state.image_error = ""

    theme_options = list(IMAGE_THEMES.keys())
    theme_index = theme_options.index(st.session_state.theme)
    selected_theme = st.selectbox("Visual theme", theme_options, index=theme_index)
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.session_state.images = []
        st.session_state.image_meta = {}
        st.session_state.image_error = ""

    if st.button("Draw new cards"):
        st.session_state.cards = draw_cards(st.session_state.spread)
        st.session_state.images = []
        st.session_state.image_meta = {}
        st.session_state.image_error = ""

    if st.button("Reset chat"):
        st.session_state.messages = []

    if st.button("Regenerate art"):
        st.session_state.images = []
        st.session_state.image_meta = {}
        st.session_state.image_error = ""

    st.divider()
    st.write("Chat model:", MODEL)
    st.write("Image model:", IMAGE_MODEL)
    if not API_KEY:
        st.warning("Set OPENAI_API_KEY in your environment or .env file.")

st.subheader("Your Spread")
st.code(cards_text(st.session_state.cards))

show_art = st.checkbox(
    "Show card art",
    value=st.session_state.show_art,
    help="Toggle on to generate DALL-E 3 art for the drawn cards.",
)
if show_art != st.session_state.show_art:
    st.session_state.show_art = show_art
    if not show_art:
        st.session_state.images = []
        st.session_state.image_meta = {}
        st.session_state.image_error = ""

if st.session_state.show_art:
    ensure_card_images()

if st.session_state.show_art and st.session_state.images:
    st.subheader("Card Art")
    art_columns = st.columns(len(st.session_state.cards))
    for column, card, image_payload in zip(art_columns, st.session_state.cards, st.session_state.images):
        with column:
            if isinstance(image_payload, (bytes, bytearray)):
                column.image(
                    image_payload,
                    caption=f"{card['name']} ({card['orientation']})",
                    use_container_width=True,
                )
            elif isinstance(image_payload, str):
                column.image(
                    image_payload,
                    caption=f"{card['name']} ({card['orientation']})",
                    use_container_width=True,
                )
            else:
                column.info("Image unavailable.")
    if st.session_state.image_error:
        st.caption(f"⚠️ {st.session_state.image_error}")
elif st.session_state.show_art and st.session_state.image_error == "missing_client":
    st.info("Add your OPENAI_API_KEY to generate tarot card artwork.")
elif st.session_state.show_art and st.session_state.image_error:
    st.warning(f"Could not generate tarot artwork right now: {st.session_state.image_error}")


# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Input → model call
if user_input := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("Shuffling the deck..."):
            answer = generate_answer(user_input)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
