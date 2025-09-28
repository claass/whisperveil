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


def generate_card_images(cards, theme_key, card_indices=None):
    theme = IMAGE_THEMES[theme_key]

    if card_indices is None:
        card_indices = list(range(len(cards)))

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

    images = [None] * len(card_indices)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(card_indices) or 1) as executor:
        future_map = {
            executor.submit(render_card, cards[idx]): position for position, idx in enumerate(card_indices)
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            image_bytes, error_msg = future.result()
            images[idx] = image_bytes
            if error_msg:
                errors.append(error_msg)

    return images, errors


def ensure_theme_images(cards, theme, *, regenerate: bool = False, prefix: str):
    if not cards:
        st.session_state[f"{prefix}_images"] = []
        st.session_state[f"{prefix}_image_error"] = ""
        st.session_state[f"{prefix}_image_meta"] = {}
        return False

    signature = card_signature(cards)
    meta = st.session_state.get(f"{prefix}_image_meta") or {}
    stored_images = list(st.session_state.get(f"{prefix}_images", []))

    if regenerate:
        stored_images = [None] * len(cards)
        meta = {}

    if len(stored_images) != len(cards):
        stored_images = (stored_images + [None] * len(cards))[: len(cards)]

    if (
        not regenerate
        and meta.get("theme") == theme
        and meta.get("signature") == signature
        and all(image is not None for image in stored_images)
    ):
        st.session_state[f"{prefix}_images"] = stored_images
        return False

    missing_indices = [idx for idx, payload in enumerate(stored_images) if payload is None]

    if not missing_indices:
        st.session_state[f"{prefix}_images"] = stored_images
        st.session_state[f"{prefix}_image_meta"] = {"theme": theme, "signature": signature}
        st.session_state[f"{prefix}_image_error"] = ""
        return False

    if not client:
        st.session_state[f"{prefix}_images"] = stored_images
        st.session_state[f"{prefix}_image_error"] = "missing_client"
        st.session_state[f"{prefix}_image_meta"] = {}
        return False

    with st.spinner("Conjuring tarot artwork..."):
        new_images, errors = generate_card_images(cards, theme, missing_indices)

    for position, card_index in enumerate(missing_indices):
        stored_images[card_index] = new_images[position]

    st.session_state[f"{prefix}_images"] = stored_images
    st.session_state[f"{prefix}_image_meta"] = {"theme": theme, "signature": signature}
    st.session_state[f"{prefix}_image_error"] = "; ".join(errors) if errors else ""
    return True


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
if "gallery_cards" not in st.session_state:
    st.session_state.gallery_cards = [
        {
            "position": card["name"],
            "name": card["name"],
            "orientation": "upright",
            "meaning": card["upright"],
        }
        for card in MAJOR_ARCANA
    ]
if "gallery_images" not in st.session_state:
    st.session_state.gallery_images = []
if "gallery_image_meta" not in st.session_state:
    st.session_state.gallery_image_meta = {"theme": None, "signature": None}
if "gallery_image_error" not in st.session_state:
    st.session_state.gallery_image_error = ""


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
        st.session_state.messages = []

    theme_options = list(IMAGE_THEMES.keys())
    theme_index = theme_options.index(st.session_state.theme)
    selected_theme = st.selectbox("Visual theme", theme_options, index=theme_index)
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.session_state.gallery_images = []
        st.session_state.gallery_image_meta = {}
        st.session_state.gallery_image_error = ""

    if st.button("Draw new cards"):
        st.session_state.cards = draw_cards(st.session_state.spread)

    if st.button("Reset chat"):
        st.session_state.messages = []

    if st.button("Regenerate art"):
        st.session_state.gallery_images = []
        st.session_state.gallery_image_meta = {}
        st.session_state.gallery_image_error = ""

    st.divider()
    st.write("Chat model:", MODEL)
    st.write("Image model:", IMAGE_MODEL)
    if not API_KEY:
        st.warning("Set OPENAI_API_KEY in your environment or .env file.")

reading_tab, gallery_tab = st.tabs(["Reading", "Gallery"])

with reading_tab:
    st.subheader("Your Spread")
    st.code(cards_text(st.session_state.cards))

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

with gallery_tab:
    st.subheader("Card Gallery")
    theme = st.session_state.theme
    cards = st.session_state.gallery_cards
    images = list(st.session_state.get("gallery_images", []))

    if len(images) != len(cards):
        images = (images + [None] * len(cards))[: len(cards)]
        st.session_state.gallery_images = images

    missing_indices = [idx for idx, payload in enumerate(images) if payload is None]

    outstanding_label = "Generate outstanding art"
    if st.button(outstanding_label, disabled=not missing_indices):
        ensure_theme_images(cards, theme, prefix="gallery")
        images = st.session_state.gallery_images
        missing_indices = [idx for idx, payload in enumerate(images) if payload is None]

    if st.button("Regenerate all art", disabled=not cards):
        ensure_theme_images(cards, theme, regenerate=True, prefix="gallery")
        images = st.session_state.gallery_images
        missing_indices = [idx for idx, payload in enumerate(images) if payload is None]

    if cards:
        columns = st.columns(3)
        active_cards = {card["name"] for card in st.session_state.cards}
        for index, (card, image_payload) in enumerate(zip(cards, images)):
            column = columns[index % len(columns)]
            with column:
                caption = f"{card['name']} ({card['orientation']})"
                if card["name"] in active_cards:
                    st.caption("Currently in your reading")
                if isinstance(image_payload, (bytes, bytearray)):
                    st.image(image_payload, caption=caption, width="stretch")
                elif isinstance(image_payload, str):
                    st.image(image_payload, caption=caption, width="stretch")
                else:
                    st.image(
                        "https://placehold.co/600x900?text=Awaiting+Art",
                        caption=caption,
                        width="stretch",
                    )
                    st.caption("Click “Generate outstanding art” to conjure this card.")
    else:
        st.info("Draw cards to begin building your gallery.")

    if st.session_state.gallery_image_error == "missing_client":
        st.info("Add your OPENAI_API_KEY to generate tarot card artwork.")
    elif st.session_state.gallery_image_error:
        st.warning(
            f"Could not generate tarot artwork right now: {st.session_state.gallery_image_error}"
        )

