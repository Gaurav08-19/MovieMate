"""
MovieMate — Gradio Web Interface
Run: python app.py
"""

import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import pandas as pd

from src.data_loader import load_movies
from src.preprocessor import preprocess_movies, add_text_representations
from src.embedder import MovieEmbedder
from src.retriever import MovieRetriever
from src.chatbot import MovieChatbot

# ---------------------------------------------------------------------------
# Bootstrap: load data, build embeddings + index, instantiate chatbot
# ---------------------------------------------------------------------------

print("=== MovieMate Initializing ===")

df_raw = load_movies()
df = preprocess_movies(df_raw)
df = add_text_representations(df)

embedder = MovieEmbedder()
embeddings = embedder.fit_transform(df["text"].tolist())

retriever = MovieRetriever(df, embedder)
retriever.build_index(embeddings)

chatbot = MovieChatbot(retriever, top_k=5)

print("=== MovieMate Ready ===\n")

# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def respond(user_message: str, history: list):
    """Gradio chat handler — called on each user message."""
    if not user_message.strip():
        return "", history

    try:
        response, retrieved = chatbot.chat(user_message)
    except Exception as e:
        response = f"Error: {e}\nMake sure ANTHROPIC_API_KEY is set."
        retrieved = []

    history.append((user_message, response))
    return "", history


def reset_chat():
    """Clear conversation history."""
    chatbot.reset()
    return [], []


def get_retrieved_movies() -> str:
    """Show the movies retrieved during the last turn."""
    movies = chatbot.get_last_retrieved()
    if not movies:
        return "No movies retrieved yet. Ask a question first!"

    lines = ["**Movies retrieved for your last query:**\n"]
    for m in movies:
        lines.append(
            f"**{m['title']}** ({m['year']}) — ⭐ {m['rating']}/10  \n"
            f"🎬 {m['director']} | 🎭 {m.get('genre', '')}  \n"
            f"🎥 {m.get('cast', '')}  \n"
            f"📝 {m.get('plot_summary', '')[:120]}...  \n"
        )
    return "\n---\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

CSS = """
.chatbot { font-size: 15px; }
.title-row { text-align: center; }
#retrieved-panel { font-size: 13px; background: #f8f9fa; padding: 12px; border-radius: 8px; }
"""

with gr.Blocks(css=CSS, title="MovieMate") as demo:

    gr.HTML("""
    <div class='title-row'>
        <h1>🎬 MovieMate</h1>
        <p style='color: #666;'>Conversational AI for Intelligent Movie Search & Recommendations</p>
    </div>
    """)

    with gr.Row():
        # Left: Chat panel
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(
                label="MovieMate Chat",
                elem_classes=["chatbot"],
                height=500,
                bubble_full_width=False,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask me anything about movies… e.g. 'Suggest sci-fi movies similar to Interstellar'",
                    label="Your message",
                    lines=2,
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            reset_btn = gr.Button("🔄 New Conversation", variant="secondary")

        # Right: Retrieved movies panel
        with gr.Column(scale=2):
            gr.Markdown("### Retrieved Movie Context")
            retrieved_md = gr.Markdown(
                value="Retrieved movies will appear here after your first message.",
                elem_id="retrieved-panel",
            )
            refresh_btn = gr.Button("Refresh Retrieved Movies")

    gr.Markdown("""
    ### 💡 Try asking:
    - *"Suggest highly rated thriller movies"*
    - *"Movies directed by Christopher Nolan"*
    - *"Recommend feel-good animated movies"*
    - *"Sci-fi movies released after 2015"*
    - *"Movies similar to Inception"*
    - *"Best drama movies with high IMDb ratings"*
    """)

    # Wire up events
    msg_state = gr.State([])

    send_btn.click(
        fn=respond,
        inputs=[msg_box, chatbot_ui],
        outputs=[msg_box, chatbot_ui],
    )
    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot_ui],
        outputs=[msg_box, chatbot_ui],
    )
    reset_btn.click(
        fn=reset_chat,
        outputs=[chatbot_ui, msg_state],
    )
    refresh_btn.click(
        fn=get_retrieved_movies,
        outputs=retrieved_md,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
