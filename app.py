"""
Gradio Web Interface for Boston School Chatbot

This script creates a web interface for your chatbot using Gradio.
You only need to implement the chat function.

Key Features:
- Creates a web UI for your chatbot
- Handles conversation history
- Provides example questions
- Can be deployed to Hugging Face Spaces

Example Usage:
    # Run locally:
    python app.py
    
    # Access in browser:
    # http://localhost:7860
"""

import json
import os
import uuid
from datetime import datetime

import gradio as gr
from src.chat import Chatbot

LOGS_DIR = "conversation_logs"


def extract_text(content) -> str:
    """Extract plain text from Gradio 5.x content, which may be a string or a list of content objects."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content)


def log_conversation(conversation_id: str, history: list, message: str, response: str):
    """
    Save the full conversation to a per-session JSON log file after each turn.

    Each file in conversation_logs/ is named <conversation_id>.json and contains
    the full conversation as plain text, ready to pass to evaluate.py via --file.

    File format:
    {
      "name": "<conversation_id>",
      "conversation": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
      ],
      "retrieved_context": ""
    }
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"{conversation_id}.json")

    turns = []
    for turn in history:
        turns.append({"role": turn["role"], "content": extract_text(turn["content"])})
    turns.append({"role": "user", "content": message})
    turns.append({"role": "assistant", "content": response})

    data = {
        "name": conversation_id,
        "timestamp": datetime.now().isoformat(),
        "conversation": turns,
        "retrieved_context": "",  # update this once RAG is implemented
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_chatbot():
    """
    Creates and configures the chatbot interface.
    """
    chatbot = Chatbot()
    conversation_id = str(uuid.uuid4())[:8]

    def chat(message, history):
        response = chatbot.get_response(message, history)
        log_conversation(conversation_id, history, message, response)
        return response

    
    
    # Create Gradio interface. Customize the interface however you'd like!
    demo = gr.ChatInterface(
        chat,
        title="BPS School Navigator",
        description="I help Boston families find the right public school for their children. Since I am a free tier chatbot, I may give a 503 error when I'm busy. If that happens, please try again a few seconds later.",
        examples=[
            "Hi, I'm looking for a school for my child who is starting kindergarten next year.",
            "What schools in Roxbury serve middle schoolers?",
            "My child has an IEP — which schools have special education programs?",
        ]
    )
    
    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()
