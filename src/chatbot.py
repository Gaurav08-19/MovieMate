"""
Core conversational chatbot for MovieMate.
Combines FAISS retrieval with Claude (Anthropic) for RAG-based responses.
"""

import os
from typing import List, Dict, Tuple, Any

import anthropic

SYSTEM_PROMPT = """You are MovieMate, an expert and enthusiastic movie recommendation assistant. You help users discover movies through natural conversation.

You have access to a curated movie database. For each user query, relevant movies are retrieved and provided to you as context. Use this context to give accurate, specific answers.

Guidelines:
- Be conversational, warm, and engaging — like talking to a knowledgeable friend who loves movies.
- Always reference specific movies from the retrieved context with their year, rating, director, and key cast.
- Explain WHY each movie fits the user's request (theme, mood, style, etc.).
- If the user asks a follow-up question, maintain context from previous turns.
- If no retrieved movies seem relevant, acknowledge it and suggest refining the query.
- Keep responses concise but informative (3-6 sentences per recommendation is ideal).
- Do NOT fabricate movie details not provided in the context.
- Format recommendations as a short numbered or bulleted list when suggesting multiple movies."""


class MovieChatbot:
    """Multi-turn conversational chatbot powered by RAG (Retrieval + Claude LLM).

    On each turn:
    1. Embed the user query and retrieve top-k relevant movies via FAISS.
    2. Inject retrieved movies as context into the LLM prompt.
    3. Call Claude with the full conversation history + context.
    4. Store clean messages (without injected context) for display.

    Args:
        retriever: A built MovieRetriever instance.
        model: Anthropic model ID to use for generation.
        top_k: Number of movies to retrieve per query.
        max_tokens: Maximum tokens for the LLM response.
    """

    def __init__(
        self,
        retriever,
        model: str = "claude-sonnet-4-6",
        top_k: int = 5,
        max_tokens: int = 1024,
    ):
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.conversation_history: List[Dict[str, str]] = []
        self._last_retrieved: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core chat method
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a user message and return (response_text, retrieved_movies).

        Args:
            user_message: The user's natural language input.

        Returns:
            Tuple of (assistant_response_string, list_of_retrieved_movie_dicts).
        """
        # 1. Retrieve relevant movies for this turn
        retrieved = self.retriever.search(user_message, k=self.top_k)
        self._last_retrieved = retrieved

        # 2. Build a context block from retrieved movies
        context = self._format_context(retrieved)

        # 3. Compose the augmented user message (sent to LLM, not stored in display history)
        augmented = (
            f"{user_message}\n\n"
            f"---\n"
            f"[Retrieved from movie database — use this to answer]\n"
            f"{context}\n"
            f"---"
        )

        # 4. Build the messages list: clean history + augmented current message
        messages = self.conversation_history + [{"role": "user", "content": augmented}]

        # 5. Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        assistant_text = response.content[0].text

        # 6. Update display history with clean (non-augmented) messages
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_text})

        return assistant_text, retrieved

    def reset(self):
        """Clear conversation history to start a fresh session."""
        self.conversation_history = []
        self._last_retrieved = []
        print("Conversation reset.")

    @property
    def history(self) -> List[Dict[str, str]]:
        """Return the clean conversation history (no injected context)."""
        return self.conversation_history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_context(self, movies: List[Dict[str, Any]]) -> str:
        """Format retrieved movies into a structured context string for the LLM."""
        if not movies:
            return "No relevant movies found in the database."

        lines = []
        for i, m in enumerate(movies, 1):
            genre = m.get("genre", "")
            cast_list = m.get("cast_list", [])
            cast = ", ".join(cast_list) if isinstance(cast_list, list) else m.get("cast", "")
            lines.append(
                f"{i}. {m['title']} ({m['year']}) — Rating: {m['rating']}/10\n"
                f"   Genre: {genre} | Director: {m['director']} | Cast: {cast}\n"
                f"   Duration: {m.get('duration_mins', '?')} mins | Language: {m.get('language', 'English')}\n"
                f"   Plot: {m.get('plot_summary', '')}"
            )
        return "\n\n".join(lines)

    def get_last_retrieved(self) -> List[Dict[str, Any]]:
        """Return the movies retrieved during the last chat turn."""
        return self._last_retrieved
