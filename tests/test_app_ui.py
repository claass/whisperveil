"""Basic end-to-end UI smoke tests for the Streamlit tarot app."""

from playwright.sync_api import Page, expect


def test_homepage_renders_core_sections(page: Page, app_base_url: str) -> None:
    page.goto(app_base_url, wait_until="networkidle")

    expect(page.get_by_role("heading", name="WhisperVeil Tarot")).to_be_visible()
    expect(page.get_by_text("For reflection and entertainment only.")).to_be_visible()
    expect(page.get_by_role("heading", name="Your Spread")).to_be_visible()
    expect(page.get_by_placeholder("Ask your question...")).to_be_visible()


def test_show_art_checkbox_prompts_for_api_key(page: Page, app_base_url: str) -> None:
    page.goto(app_base_url, wait_until="networkidle")
    page.mouse.wheel(0, 2000)

    checkbox = page.get_by_role("checkbox", name="Show card art")
    checkbox.scroll_into_view_if_needed()
    checkbox.evaluate("node => node.click()")

    expect(
        page.get_by_text("Add your OPENAI_API_KEY to generate tarot card artwork.")
    ).to_be_visible()
