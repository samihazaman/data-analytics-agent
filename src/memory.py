"""Conversation memory — stores chat history and supports summarization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


@dataclass
class ConversationMemory:
    """
    Sliding-window conversation memory with optional summarization.

    Keeps the last `window_size` turns in full, and compresses older
    turns into a running summary so the context window stays manageable.
    """

    window_size: int = 10
    _history: list[dict[str, str]] = field(default_factory=list)
    _summary: str = ""

    def add_user(self, message: str) -> None:
        """Record a user turn."""
        self._history.append({"role": "user", "content": message})

    def add_assistant(self, message: str) -> None:
        """Record an assistant turn."""
        self._history.append({"role": "assistant", "content": message})

    def get_messages(self) -> list[dict[str, str]]:
        """Return the message list to inject into the next LLM call."""
        messages: list[dict[str, str]] = []

        if self._summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{self._summary}",
            })

        # Only keep the most recent window_size turns
        messages.extend(self._history[-self.window_size :])
        return messages

    def maybe_summarize(self, client: OpenAI, model: str = "gpt-4o-mini") -> None:
        """
        If history exceeds the window, compress the oldest half into the summary.
        Call this after each completed turn.
        """
        if len(self._history) <= self.window_size:
            return

        cutoff = len(self._history) - self.window_size
        to_compress = self._history[:cutoff]
        self._history = self._history[cutoff:]

        turns_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in to_compress
        )
        prompt = (
            "Summarize the following conversation turns concisely, "
            "preserving key facts, dataset names, and analysis results:\n\n"
            f"{turns_text}"
        )
        if self._summary:
            prompt = f"Existing summary:\n{self._summary}\n\nNew turns to add:\n{turns_text}\n\nUpdate the summary."

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        self._summary = response.choices[0].message.content.strip()

    def clear(self) -> None:
        """Reset all history and summary."""
        self._history = []
        self._summary = ""
