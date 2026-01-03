# OllaTalk
# LlamaFlow: Async Ollama CLI

A lightweight, asynchronous, and streaming terminal client for [Ollama](https://ollama.com/), built with Python. 

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## 📖 Overview

This project provides a robust, "advanced" implementation of a chat interface for local LLMs. Unlike basic scripts, this engine handles **asynchronous I/O**, **token streaming**, and **persistent conversation context** without blocking the main thread.

It is designed to be modular: the `Engine` logic is separated from the `Configuration` and the `Main` application loop.

## ✨ Key Features

* **⚡ Asynchronous Core:** Built on `asyncio` for non-blocking performance.
* **🌊 Real-time Streaming:** Displays tokens instantly as they are generated (typing effect).
* **🧠 Context Awareness:** Automatically manages conversation history so the model "remembers" previous turns.
* **🛡️ Robust Error Handling:** Gracefully manages API connection failures.
* **⚙️ Configurable:** Easy to swap models (`llama3`, `mistral`, etc.) and system prompts.

## 🚀 Prerequisites

1.  **Python 3.8+** installed.
2.  **Ollama** installed and running.
    * Download from [ollama.com](https://ollama.com)
3.  Pull the model you intend to use (default is `llama3.2`):
    ```bash
    ollama pull llama3.2
    ```

## 🛠️ Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/Shravanrp/OllaTalk.git]
    cd llamaflow
    ```

2.  Install the dependencies:
    ```bash
    pip install ollama
    ```

## 💻 Usage

Run the script directly from your terminal:

```bash
python main.py
