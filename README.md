<div align="center">

# 📚 DocPilot AI

### Your Intelligent AI Study & Document Assistant

**Understand Documents · Learn Smarter · Remember Longer**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![AI Providers](https://img.shields.io/badge/AI-Multi--Provider-7C3AED)](#-configure-cloud-ai-providers)
[![RAG](https://img.shields.io/badge/RAG-Document--Aware-0F9D58)](#-document-intelligence--rag)
[![Vision](https://img.shields.io/badge/Vision-Multimodal-2563EB)](#-vision-solver)
[![Exports](https://img.shields.io/badge/Exports-PDF%20%7C%20DOCX%20%7C%20PPTX%20%7C%20XLSX-F59E0B)](#-export-and-download)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Python-64748B)](#-installation--setup)

[▶️ Watch the DocPilot AI Tutorial](https://youtu.be/H4GxjIoe66I) · [🌐 Open the Live Web Application](https://docpilot-ai-1.onrender.com/)

</div>

---

## 📌 Table of Contents

- [🚀 About DocPilot AI](#-about-docpilot-ai)
- [✨ Features](#-features)
- [🧱 Architecture](#-architecture)
- [📂 Complete Project Structure](#-complete-project-structure)
- [💻 Requirements](#-requirements)
- [⚙️ Installation and Setup](#️-installation--setup)
- [▶️ Run DocPilot AI](#️-run-docpilot-ai)
- [🔑 Configure Cloud AI Providers](#-configure-cloud-ai-providers)
- [🖥️ Use Local AI Models](#️-use-local-ai-models)
- [📂 Document Intelligence and RAG](#-document-intelligence--rag)
- [👁️ Vision Solver](#️-vision-solver)
- [🧠 Study Tools](#-study-tools)
- [📤 Export and Download](#-export-and-download)
- [🌐 LAN Access](#-lan-access)
- [☁️ Render Deployment](#️-render-deployment)
- [🗃️ Data, Storage and Backups](#️-data-storage-and-backups)
- [🔌 API Endpoint Reference](#-api-endpoint-reference)
- [🧪 Testing and Verification](#-testing-and-verification)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [🔐 Security Notes](#-security-notes)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🚀 About DocPilot AI

**DocPilot AI** is a Python-powered AI study and document assistant. It brings document understanding, AI conversations, Retrieval-Augmented Generation (RAG), image understanding, study-material generation, and multi-format exports into one application.

The supplied project runs as a single Python application. Its built-in web interface is served by the Python server, so a separate Node.js, npm, Vite, or frontend build process is not required.

DocPilot AI supports two ways to use AI:

1. **Cloud AI providers** — enter your own provider API key through DocPilot's Provider Settings interface.
2. **Local AI models** — connect DocPilot to a local inference server such as Ollama or LM Studio. Local inference uses your own computer's hardware.

The application does not require users to create a `.env` file to enter provider API keys. Provider keys are configured in the application's UI.

## ✨ Features

| Feature | Description |
|---|---|
| 📖 **Study Copilot** | Conversational AI assistant for learning, explanations, and follow-up questions. |
| 📂 **Document Understanding** | Upload supported documents and extract their content. |
| 🔎 **RAG Document Retrieval** | Retrieve relevant document chunks to provide source context to answers. |
| 🤖 **Multiple AI Providers** | Configure Gemini, Groq, OpenRouter, OpenAI, or Anthropic in the UI. |
| 🖥️ **Local AI Inference** | Connect to Ollama, LM Studio, or compatible local inference endpoints. |
| 👁️ **Vision Solver** | Analyze images using a model that supports image input. |
| 🧭 **Intent-Aware Workflows** | Identify user requests and route study tasks to relevant workflows. |
| 📝 **Quiz Generation** | Generate practice questions from study context. |
| 🃏 **Recall Cards** | Create flashcards for active recall and revision. |
| 📑 **Cheat Sheets** | Generate condensed notes and regenerate individual sections. |
| 📤 **Document Exports** | Create PDF, DOCX, PPTX, and XLSX files. |
| 💬 **Conversation Sessions** | Store sessions and messages in the local application database. |

> Available models, vision support, usage limits, pricing, and free quotas depend on the provider. A model that only supports text cannot perform image analysis.

## 🧱 Architecture

```text
┌──────────────────────────────────────────────┐
│                 Web Browser                  │
│             frontend/index.html              │
└──────────────────────┬───────────────────────┘
                       │ Same-origin HTTP
                       ▼
┌──────────────────────────────────────────────┐
│                    main.py                   │
│              Application entry point         │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│             app/api/handlers.py              │
│        Threaded HTTP server and API logic    │
└──────────┬──────────────┬────────────────────┘
           │              │
           ▼              ▼
┌──────────────────┐  ┌────────────────────────┐
│  AI / Study      │  │   Document / Vision    │
│  Services        │  │   / Export Services   │
└─────────┬────────┘  └────────────┬───────────┘
          │                        │
          └───────────┬────────────┘
                      ▼
          ┌────────────────────────┐
          │  app/database/db.py    │
          │  SQLite + app records  │
          └────────────────────────┘
                      │
                      ▼
          ┌────────────────────────┐
          │ .data / storage / logs │
          │ Runtime application   │
          │ files and outputs     │
          └────────────────────────┘
```

### Runtime facts

- `main.py` imports and starts the server implemented in `app/api/handlers.py`.
- The server uses Python's `http.server` and `ThreadingMixIn`.
- The frontend is served from `frontend/index.html`.
- The application uses SQLite through `app/database/db.py`.
- The default local port is `8000`.
- The default local bind address is `127.0.0.1`.
- `HOST`, `PORT`, `DOCPILOT_DATA_DIR`, and `DOCPILOT_NO_BROWSER` are supported runtime settings.
- The project includes FastAPI/Uvicorn in `requirements.txt`, but the current `main.py` launch path uses the custom threaded HTTP server. Start the supplied application with `python main.py`; do not use `uvicorn main:app` for this source version.

---

## 📂 Complete Project Structure

The GitHub repository and extracted project folder should be named **`DocPilot-AI`**.

```text
DocPilot-AI/
│
├── main.py
│   └── Main entry point. Imports the application handler and starts
│       the DocPilot AI server when executed with `python main.py`.
│
├── requirements.txt
│   └── Python dependencies used by the application:
│       AI provider clients, HTTP utilities, encryption, document
│       extraction, image processing, and export-generation libraries.
│
├── setup.bat
│   └── Windows setup helper. Checks Python, creates .venv,
│       installs requirements, creates local runtime folders,
│       and attempts unittest discovery if a tests folder exists.
│
├── run.bat
│   └── Windows launcher. Activates .venv when available and
│       starts the application using `python main.py`.
│
├── README.md
│   └── Project documentation, setup instructions, provider setup,
│       local-model usage, deployment notes, and troubleshooting.
│
├── app/
│   │
│   ├── config.py
│   │   └── Central path and runtime configuration:
│   │       project root, frontend root, data root, HOST, and PORT.
│   │       Supports DOCPILOT_DATA_DIR for a custom data location.
│   │
│   ├── api/
│   │   ├── handlers.py
│   │   │   └── Main threaded HTTP server and request handler.
│   │   │       Serves the frontend, dispatches API requests,
│   │   │       coordinates chat turns, document uploads,
│   │   │       provider/model operations, vision, study tools,
│   │   │       exports, and downloads.
│   │   │
│   │   └── routes.py
│   │       └── Route/controller compatibility layer that exposes
│   │           selected handler functions.
│   │
│   ├── database/
│   │   └── db.py
│   │       └── SQLite persistence, schema initialization and
│   │           migrations, sessions, messages, documents, chunks,
│   │           settings, provider status, and encrypted credential
│   │           vault operations.
│   │
│   └── services/
│       ├── ai_provider.py
│       │   └── Central AI provider manager. Handles provider IDs,
│       │       saved credentials, provider verification, model
│       │       discovery, model metadata, model selection,
│       │       generation requests, retry handling, usage/latency
│       │       information, and local inference connections.
│       │
│       ├── documents_service.py
│       │   └── Reads supported documents, extracts text, structures
│       │       content, splits text into chunks, and prepares
│       │       document content for retrieval.
│       │
│       ├── rag_service.py
│       │   └── Retrieves and ranks relevant document chunks for
│       │       a user's query.
│       │
│       ├── intent_classifier.py
│       │   └── Classifies user intent and identifies study-related
│       │       actions and contextual follow-up requests.
│       │
│       ├── study_service.py
│       │   └── Study workflows including quizzes, flashcards/
│       │       Recall Cards, Cheat Sheets, and section regeneration.
│       │
│       ├── vision_service.py
│       │   └── Image validation, image-category classification,
│       │       vision capability checks, and image analysis.
│       │
│       └── export_service.py
│           └── Generates downloadable PDF, DOCX, PPTX, and XLSX
│               files and validates generated output.
│
├── frontend/
│   └── index.html
│       └── Complete browser UI: layout, styling, settings,
│           provider configuration, model selection, chat,
│           document controls, Vision Solver, study tools,
│           and frontend API calls.
│
├── .data/                    [created/used at runtime]
│   ├── docpilot.db           SQLite database
│   ├── uploads/              Uploaded documents
│   └── exports/              Generated files
│
├── storage/                  [created/used at runtime]
│   ├── uploads/              Upload-related files
│   ├── extracted/            Extracted document content
│   ├── indexes/              Document index data
│   ├── exports/              Export-related files
│   ├── temp/                 Temporary processing files
│   └── backups/              Backup-related files
│
└── logs/                     [created at runtime]
    └── Application log files
```
---

## 💻 Requirements

### Local Windows requirements

- Windows 10 or Windows 11
- Python **3.10, 3.11, or 3.12 (64-bit)** recommended by the supplied setup script
- Internet connection for installing packages and using cloud providers
- Modern browser (Chrome, Edge, Firefox, etc.)
- Enough disk space for Python dependencies, uploaded files, and generated exports

### Optional requirements

- **Ollama** for local model inference
- **LM Studio** for local OpenAI-compatible inference
- A downloaded local model that fits your computer's RAM/VRAM
- An API account/key for a cloud provider

You do not need Node.js, npm, Vite, or a separate frontend server for the supplied application.

---

## ⚙️ Installation & Setup

### 1. Download the repository

Clone using Git:

```bash
git clone https://github.com/darshanhv296/Docpilot-ai.git
```

Or download the repository ZIP from GitHub and extract it.

### 2. Rename/open the project folder

The extracted project root should be:

```text
DocPilot-AI
```

Open Command Prompt or PowerShell inside that folder.

Example:

```powershell
cd DocPilot-AI
```

### 3. Run the Windows setup script

Double-click:

```text
setup.bat
```

The script:
1. Checks whether Python is available.
2. Creates `.venv` if it does not exist.
3. Activates the virtual environment.
4. Upgrades pip.
5. Installs `requirements.txt`.
6. Creates local data/storage/log directories.

### 4. Manual installation (alternative)

If you prefer not to use `setup.bat`, run:

```powershell
# Create virtual environment
python -m venv .venv

# Activate in PowerShell
.\.venv\Scripts\Activate.ps1

# Or activate in Command Prompt
.venv\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

---

## ▶️ Run DocPilot AI

### Option A — Windows launcher

Double-click:

```text
run.bat
```

### Option B — Terminal

From the `DocPilot-AI` folder:

```bash
python main.py
```

The application normally opens:

```text
http://localhost:8000
```

If the browser does not open automatically, copy the address printed in the terminal into your browser.

### Stop the application

Focus the terminal window and press:

```text
Ctrl + C
```

Keep the terminal open while using the application. Closing it stops the local server.

---

## 🔑 Configure Cloud AI Providers

DocPilot AI has a built-in **Provider Settings / Config & Keys** interface. You obtain an API key from the provider's official dashboard, then paste it into DocPilot's UI.

**You do not need to put provider API keys in `main.py`, `config.py`, `requirements.txt`, frontend JavaScript, or a `.env` file.**

### Providers available in DocPilot

| Provider | Provider ID | Get an API key |
|---|---|---|
| Google Gemini | `gemini` | [Google AI Studio — API Keys](https://aistudio.google.com/app/apikey) |
| Groq | `groq` | [Groq Console — API Keys](https://console.groq.com/keys) |
| OpenRouter | `openrouter` | [OpenRouter — Keys](https://openrouter.ai/settings/keys) |
| OpenAI | `openai` | [OpenAI Platform — API Keys](https://platform.openai.com/api-keys) |
| Anthropic Claude | `anthropic` | [Anthropic Console — API Keys](https://platform.claude.com/settings/keys) |
| Local inference | `local` | No cloud API key normally required; configure a local model server. |

### A. Google Gemini API key

**Official link:** [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Steps:
1. Open Google AI Studio.
2. Sign in with your Google account.
3. Open the API Keys page.
4. Create an API key (or select an existing project/key).
5. Copy the key and keep it private.
6. Open DocPilot AI.
7. Open **Settings → Provider / Config & Keys**.
8. Select **Gemini**.
9. Paste the key into the API key field.
10. Click **Verify & Fetch Models**.
11. Choose an available model and save/apply the provider configuration.

Official getting-started guide: [Gemini API documentation](https://ai.google.dev/gemini-api/docs/get-started).

### B. Groq API key

**Official link:** [https://console.groq.com/keys](https://console.groq.com/keys)

Steps:
1. Sign in or create a Groq Console account.
2. Open API Keys.
3. Create a new key.
4. Copy it securely.
5. In DocPilot, open **Settings → Provider / Config & Keys**.
6. Select **Groq**.
7. Paste the key.
8. Click **Verify & Fetch Models**.
9. Select an available model and save/apply.

Official guide: [Groq Quickstart](https://console.groq.com/docs/quickstart).

### C. OpenRouter API key

**Official link:** [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)

Steps:
1. Sign in to OpenRouter.
2. Open the Keys page.
3. Create a key.
4. Copy it securely. The full secret may only be displayed at creation.
5. In DocPilot, open **Settings → Provider / Config & Keys**.
6. Select **OpenRouter**.
7. Paste the key.
8. Click **Verify & Fetch Models**.
9. Select a model returned by the application and save/apply.

Official guide: [OpenRouter Developer Platform](https://openrouter.ai/developers).

OpenRouter provides access to models from multiple providers. Model availability, context limits, pricing, and free-model availability vary. Check the model's listing and your account's usage/credit balance before relying on it.

### D. OpenAI API key

**Official link:** [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

Steps:
1. Sign in to the OpenAI API Platform.
2. Select or create the appropriate API project.
3. Open API Keys.
4. Create a new secret key.
5. Copy it immediately and store it privately.
6. In DocPilot, open **Settings → Provider / Config & Keys**.
7. Select **OpenAI**.
8. Paste the key.
9. Click **Verify & Fetch Models**.
10. Select an available model and save/apply.

Official guide: [OpenAI API key documentation](https://developers.openai.com/api/docs/guides/api-keys).

**Billing note:** ChatGPT subscriptions and OpenAI API billing are separate. Having a ChatGPT plan does not automatically provide API credits. Check API billing and usage in the OpenAI Platform before making requests.

### E. Anthropic Claude API key

**Official link:** [https://platform.claude.com/settings/keys](https://platform.claude.com/settings/keys)

Steps:
1. Sign in to the Claude Console.
2. Open API Keys.
3. Create a key and give it a recognizable name.
4. Copy the key securely. The full key is shown at creation.
5. In DocPilot, open **Settings → Provider / Config & Keys**.
6. Select **Anthropic**.
7. Paste the key.
8. Click **Verify & Fetch Models**.
9. Select an available model and save/apply.

Official instructions: [Get your Claude API key](https://platform.claude.com/docs/en/get-api-key).

### Common provider setup workflow

The same workflow applies to the cloud providers:

```text
Provider's official website
          │
          ▼
Create an API key
          │
          ▼
Copy the secret key
          │
          ▼
Open DocPilot AI
          │
          ▼
Settings → Provider / Config & Keys
          │
          ▼
Select the matching provider
          │
          ▼
Paste API key
          │
          ▼
Verify & Fetch Models
          │
          ▼
Choose model → Save / Apply
          │
          ▼
Start a test conversation
```

### How to confirm that the key works

After saving a provider:
1. Check the provider status shown in DocPilot.
2. Run model discovery again if the model list is empty.
3. Select a model that is shown as available.
4. Start a short test conversation, such as: `Explain what RAG means in two sentences.`
5. If it fails, check the provider dashboard for key status, billing/credits, usage limits, and model access.

### API usage, billing, and free-tier notes

- An API key is an authentication credential; it is not automatically free usage.
- Some providers offer limited free access or selected free models, subject to changing quotas and policies.
- Other providers require billing setup or prepaid credits.
- Usage can be charged based on input/output tokens, requests, model, or other provider-specific rules.
- Rate limits and free quotas can change.
- Set provider-side spending/usage limits where available.
- Monitor usage on the provider's official dashboard.
- Never assume a model is free solely because its name includes `free`; verify the current provider listing and account terms.

---

## 🖥️ Use Local AI Models

DocPilot's Local provider is designed to connect to a local inference server. The UI describes local connections for **Ollama, LM Studio, or vLLM-compatible endpoints**.

Local inference means the model runs on a computer or server you control. It does not mean the model is hosted by DocPilot.

### Option 1 — Ollama (recommended for a straightforward local setup)

#### 1. Install Ollama

Download and install:

[🌐 Ollama Official Website](https://ollama.com/download)

#### 2. Download a model

Open Command Prompt or PowerShell:

```bash
ollama pull llama3.2
```

Check downloaded models:

```bash
ollama list
```

You can choose another model supported by your hardware. Larger models generally need more RAM/VRAM and may respond more slowly.

#### 3. Confirm Ollama is running

Ollama normally exposes its local API at:

```text
http://localhost:11434
```

Check the installed model list:

```bash
ollama list
```

#### 4. Connect Ollama to DocPilot

1. Start DocPilot AI.
2. Open **Settings → Provider / Config & Keys**.
3. Select **Local**.
4. Enable the local inference mode if the UI presents the toggle.
5. Set the Local Endpoint Base URL to:

   ```text
   http://localhost:11434
   ```

6. Click **Verify & Fetch Models**.
7. Select one of the discovered models.
8. Save/apply the selection.
9. Send a test message.

For Ollama on the same computer as DocPilot, `localhost:11434` is normally the correct endpoint.

### Option 2 — LM Studio

1. Download LM Studio: [https://lmstudio.ai/](https://lmstudio.ai/)
2. Install and open LM Studio.
3. Download a model compatible with your computer.
4. Load the model.
5. Start LM Studio's local server from its Developer/Local Server interface.
6. Note the server's displayed base URL. The common default is:

   ```text
   http://localhost:1234/v1
   ```

7. In DocPilot, select **Local**.
8. Enter the LM Studio server URL accepted by the Local Endpoint Base URL field.
9. Click **Verify & Fetch Models**.
10. Select a discovered model and test it.

Use the exact endpoint shown by LM Studio. Do not append or remove `/v1` unless the server configuration or DocPilot UI requires it.

Official website: [LM Studio](https://lmstudio.ai/).

### Option 3 — vLLM or another compatible endpoint

If using vLLM or another OpenAI-compatible inference server:
1. Start the server using its own installation and configuration guide.
2. Confirm the server is reachable from the machine running DocPilot.
3. Copy its OpenAI-compatible base URL.
4. Select **Local** in DocPilot.
5. Enter the reachable endpoint.
6. Run model discovery/verification.
7. Select the returned model and test a prompt.

Official vLLM documentation: [https://docs.vllm.ai/](https://docs.vllm.ai/).

### Cloud AI vs local AI

| Cloud AI provider | Local AI model |
|---|---|
| Inference runs on provider infrastructure. | Inference runs on your computer/server. |
| Requires a provider API key. | Usually does not require a cloud API key. |
| Internet is required for requests. | Can work offline after model download, if all local dependencies are available. |
| Provider quotas, billing, and rate limits may apply. | Speed depends on CPU, GPU, RAM, VRAM, model size, and quantization. |
| Provider hosts and updates the model. | You download, manage, and update the model. |

### Local model performance tips

- Start with a smaller model that fits available RAM/VRAM.
- Close unnecessary GPU-heavy applications.
- Avoid loading multiple large models simultaneously.
- Use a model that supports the required task. Image analysis needs vision capability.
- Keep the local inference server running while using DocPilot.
- If model discovery fails, verify the local server URL and its status first.

### Important: local models and a hosted DocPilot website

`localhost` always means **the machine on which the DocPilot backend is running**.

- When DocPilot and Ollama run on your laptop, `http://localhost:11434` points to your laptop's Ollama service.
- When DocPilot is deployed on Render, `localhost:11434` points to the Render container—not your laptop.
- A hosted DocPilot server cannot automatically access Ollama running on a visitor's private computer.
- To use local inference with a hosted deployment, you need a deliberately configured, secure, reachable inference endpoint or a separate local DocPilot installation.

Do not expose Ollama, LM Studio, or vLLM directly to the public internet without authentication and network protections.

---

## 📂 Document Intelligence & RAG

DocPilot processes supported documents so that relevant extracted text can be used in AI conversations.

### Supported document extraction paths in the source

| File type | Extraction path |
|---|---|
| PDF | PDF text extraction |
| DOCX | Word document extraction |
| PPTX | PowerPoint text extraction |
| XLSX | Spreadsheet extraction |
| CSV | CSV text extraction |
| IPYNB | Jupyter Notebook extraction |
| Images | Image-document extraction path |

Actual extraction depends on file structure and content. Scanned PDFs or images may require OCR capabilities; do not assume every scanned file will produce perfect text.

### Typical document workflow

1. Open a DocPilot conversation.
2. Use the document upload control.
3. Select a supported file.
4. Wait for the upload and processing step to finish.
5. Ask a question about the uploaded document.
6. Ask follow-up questions using the same document context.
7. Use the study tools to create revision material from the available context.

### What RAG does

Retrieval-Augmented Generation combines:
- **Retrieval:** find relevant chunks from the indexed document content.
- **Augmentation:** add those chunks as context for the AI request.
- **Generation:** ask the selected AI model to produce an answer using that context.

RAG helps the model answer with reference to uploaded content, but it does not guarantee perfect extraction or factual accuracy. Verify important information against the original document.

---

## 👁️ Vision Solver

Vision Solver is intended for image understanding through a provider/model that supports image input.

### Typical workflow

1. Open **Vision Solver**.
2. Upload or select an image using the interface.
3. Choose the available analysis/solve action.
4. Run the request.
5. Review the model's explanation.

Example image types include:
- Mathematical equations
- Physics questions
- Chemistry reactions
- Circuit diagrams
- Engineering diagrams
- Graphs and charts
- Handwritten notes
- Document screenshots
- Code or terminal screenshots

**Important:** Vision requires a compatible multimodal model. A text-only model cannot inspect an image. Provider model capabilities and image limits vary.

---

## 🧠 Study Tools

### 📝 Quiz Generator

Generate practice questions from the active topic or available document context. Use the quiz interface to review and answer questions.

### 🃏 Recall Cards

Generate question-and-answer flashcards for active recall. Try answering before revealing the answer.

### 📑 Cheat Sheets

Generate condensed revision notes from the active topic or source material. Where available, regenerate an individual section to refine it.

### 📖 Study Copilot

Ask for:
- Concept explanations
- Step-by-step breakdowns
- Topic summaries
- Comparisons
- Examples
- Clarifications and follow-up questions

The quality of study content depends on the model, prompt, and source context. Always verify high-stakes academic, technical, or factual material.

---

## 📤 Export and Download

DocPilot's export service supports the following formats:

| Format | Purpose | Python library |
|---|---|---|
| PDF | Portable, print-ready documents | ReportLab |
| DOCX | Editable Word documents | python-docx |
| PPTX | Editable PowerPoint presentations | python-pptx / OpenXML fallback |
| XLSX | Editable Excel workbooks | openpyxl |

### Export workflow

1. Generate or prepare content in DocPilot.
2. Use the available export/download controls.
3. Choose the required format.
4. Wait for file generation and validation.
5. Download the generated file.
6. Open the file in a compatible desktop application to verify formatting.

Large documents and complex layouts may take longer to generate.

---

## 🌐 LAN Access

By default, DocPilot uses `127.0.0.1`, which makes it available only from the same computer.

To access DocPilot from another device on the same trusted local network, start the server with `HOST=0.0.0.0`.

### Windows Command Prompt

```bat
set HOST=0.0.0.0
set PORT=8000
python main.py
```

### Find your computer's LAN IP

```bat
ipconfig
```

Find the active network adapter's IPv4 Address, for example:

```text
192.168.1.10
```

On another device connected to the same network, open:

```text
http://192.168.1.10:8000
```

Replace the example IP with your computer's actual LAN IP.

**Network notes:**
- `127.0.0.1` always means the device itself.
- `0.0.0.0` is a server bind address, not a browser URL.
- Use the computer's LAN IPv4 address from the other device.
- Windows Firewall may need an inbound rule for port 8000 on a trusted Private network.
- Guest Wi-Fi or client isolation may block device-to-device access.
- Do not expose the development server to an untrusted public network.

---

## ☁️ Render Deployment

The application is currently a single Python server that serves both the frontend and API. A first Render deployment can therefore use **one Render Web Service** rather than splitting the frontend and backend.

### Suggested Render service settings

| Setting | Value |
|---|---|
| Service type | Web Service |
| Runtime | Python |
| Build command | `pip install -r requirements.txt` |
| Start command | `python main.py` |
| `HOST` | `0.0.0.0` |
| `PORT` | Use Render's assigned `PORT` |
| `DOCPILOT_NO_BROWSER` | `1` |
| `DOCPILOT_DATA_DIR` | Path to a mounted persistent disk |

### Before deploying

The current local configuration defaults to `HOST=127.0.0.1`. For Render, the process must bind to `0.0.0.0`. The current handler reads `PORT` but its `HOST` default comes from `app/config.py`; configure `HOST=0.0.0.0` in Render's environment settings or update the deployment configuration appropriately.

Also verify:
- The data directory is on persistent storage.
- SQLite database files survive restarts and redeployments.
- Uploaded documents and generated exports use persistent storage.
- The credential vault's encryption key remains stable and is backed up.
- The hosted service has appropriate authentication and authorization before public multi-user access.
- Local Ollama endpoints are not assumed to be available from Render.

### Important deployment and security limitation

The presence of a provider-key UI does not by itself make a deployment safe for unrelated public users. Before opening the application to the public, implement and test authentication, per-user document/session isolation, authorization for file downloads, request limits, secure credential handling, and persistent encryption-key management.

A public deployment must not expose one user's provider credentials, documents, conversations, or generated files to another user.

Render documentation: [https://render.com/docs](https://render.com/docs).

---

## 🗃️ Data, Storage and Backups

The project configuration defines:
- `.data/` as the default source-run data directory.
- `DOCPILOT_DATA_DIR` as an optional override.
- `docpilot.db` as the SQLite database file.
- `uploads/` and `exports/` under the data directory.
- `storage/` and `logs/` as additional application runtime directories.

### Backup procedure

1. Stop DocPilot AI.
2. Copy the `.data/` directory to a secure backup location.
3. Back up any required files under `storage/`.
4. Securely preserve the credential vault's matching encryption-key material.
5. Restart DocPilot AI.

The encrypted credential database and its matching encryption-key material must be preserved together. Losing the encryption key can make previously saved provider credentials unrecoverable.

Do not delete `.data/`, vault key files, or `storage/` folders as a routine troubleshooting step.

---

## 🔌 API Endpoint Reference

The current HTTP handler contains the following application routes. These are internal DocPilot application endpoints, not public third-party provider APIs.

| Endpoint | Purpose |
|---|---|
| `/health` and `/api/health` | Health/status response |
| `/api/chat` | Chat request |
| `/api/chat/stream` | Streaming chat path |
| `/api/chat/sessions` | Conversation/session operations |
| `/api/documents` | Document operations |
| `/api/documents/upload` | Upload and process documents |
| `/api/documents/clear` | Clear document data through the application workflow |
| `/api/files` | File operations |
| `/api/files/download/{zip_fname}` | Generated ZIP download |
| `/api/export` | Generate exports |
| `/api/models/discover` | Discover provider models |
| `/api/models/live` | Live model information |
| `/api/providers/status` | Provider status |
| `/api/providers/recommendations` | Provider/model recommendations |
| `/api/settings` | Application settings |
| `/api/profile` | Profile operations |
| `/api/user/profile` | User profile operations |
| `/api/memories` | Memory operations |
| `/api/intent/status` | Intent classifier status |
| `/api/rag/search` | Document retrieval search |
| `/api/vector-store/status` | Vector/index status |
| `/api/vision/categories` | Vision categories |
| `/api/vision/classify` | Image classification |
| `/api/vision/analyze` | Image analysis |
| `/api/vision/solve` | Vision solving |
| `/api/vision/compatibility` | Vision model compatibility |
| `/api/study/flashcards` | Recall-card generation |
| `/api/study/quizzes` | Quiz generation |
| `/api/study/quizzes/submit` | Quiz submission |
| `/api/study/cheatsheet` | Cheat-sheet generation |
| `/api/study/cheatsheet/section` | Cheat-sheet section regeneration |
| `/api/code/zip` | Code ZIP operation |

For exact request bodies and response schemas, inspect the matching route branch in `app/api/handlers.py` and the corresponding frontend request in `frontend/index.html`.

---

## 🧪 Testing and Verification

The supplied `setup.bat` attempts to run:

```bat
python -m unittest discover -s tests -p "test_*.py"
```

The supplied source archive does not include a `tests/` directory. Therefore, this command may discover no tests. Do not interpret setup completion as proof that all features passed automated tests.

### Recommended manual verification checklist

- [ ] `python main.py` starts without an exception.
- [ ] Browser opens `http://localhost:8000`.
- [ ] `/api/health` responds.
- [ ] Provider settings opens correctly.
- [ ] A cloud provider key can be entered and verified.
- [ ] Model discovery returns available models.
- [ ] A short cloud-model chat succeeds.
- [ ] Ollama is discovered when running locally.
- [ ] A local-model chat succeeds.
- [ ] A supported document uploads and extracts text.
- [ ] RAG returns relevant content for a document question.
- [ ] Vision works with an image-capable model.
- [ ] Quiz generation works.
- [ ] Recall Cards generation works.
- [ ] Cheat Sheet generation works.
- [ ] PDF export opens correctly.
- [ ] DOCX export opens correctly.
- [ ] PPTX export opens correctly.
- [ ] XLSX export opens correctly.
- [ ] Data remains available after a normal application restart.

---

## 🛠️ Troubleshooting

### 1. Python is not recognized

Install Python 3.10–3.12 (64-bit) and enable **Add Python to PATH** during installation. Reopen the terminal after installation.

### 2. Dependency installation fails

Activate `.venv` and run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Read the first package installation error in the terminal. Some packages may need a compatible Python version or system build tools.

### 3. Port 8000 is already in use

The application has a port-selection helper. Read the terminal startup output to identify the actual port. Open the printed URL.

### 4. Provider key verification fails

- Confirm you selected the correct provider tab.
- Check that the key was copied completely.
- Confirm the key is active and has the required API access.
- Check provider usage, billing/credits, and rate limits.
- Run **Verify & Fetch Models** again.
- Select a model returned by model discovery.
- Check the terminal/log output for the provider error.

### 5. Provider model list is empty

- Confirm provider verification succeeded.
- Check internet access.
- Confirm the provider account has access to the model API.
- Check whether the provider has changed or retired the model.
- Retry model discovery after checking provider status.

### 6. Local model is not detected

- Confirm Ollama or LM Studio is running.
- For Ollama, run `ollama list`.
- Confirm the endpoint URL.
- Confirm the local server is reachable from the same machine that runs DocPilot.
- Click **Verify & Fetch Models** again.

### 7. Local model responses are slow

Local inference speed depends on model size, quantization, RAM, GPU/VRAM, CPU, context length, and other running processes. Try a smaller model and close other GPU-intensive applications.

### 8. Vision Solver fails

Select a model that supports image input. Check provider/model image capability, image size/type, provider limits, and account status.

### 9. Document answers are incomplete

Confirm document processing finished. Ask a focused question, ensure the correct document/session is active, and compare the answer with the original source. Poor scans or image-only PDFs may not extract usable text.

### 10. Export generation fails

Check terminal/log output, available disk space, and write permissions for the data/export directory. Confirm the generated file exists and opens in the target application.

### 11. Saved API key cannot be decrypted

The credential vault relies on encryption-key material. Restore the database and matching key backup together. Do not delete or regenerate the key file without understanding the migration/recovery consequences.

---

## 🔐 Security Notes

- Treat every provider API key as a password.
- Never commit API keys, `.data/`, credential-vault files, uploads, exports, logs, `.venv/`, or `__pycache__/` to GitHub.
- The application UI is the intended place to enter provider keys for local use.
- Do not expose a shared deployment without authentication and per-user authorization.
- Protect file upload and download routes.
- Back up the database and matching credential encryption-key material securely.
- Use HTTPS for public deployments.
- Do not expose local inference endpoints directly to the public internet.
- Review each provider's privacy, retention, pricing, and acceptable-use terms before sending sensitive documents.

### Suggested `.gitignore`

Create a `.gitignore` file in the project root with:

```gitignore
# Python virtual environment
.venv/
venv/
env/

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Local secrets and environment configuration
.env
.env.*
!.env.example

# DocPilot runtime data
.data/
data/
storage/
logs/

# Generated files
*.db
*.sqlite
*.sqlite3
*.log
*.tmp

# OS/editor files
.DS_Store
Thumbs.db
.vscode/
.idea/
```

Review this list before committing. If you intentionally need to version a non-secret example configuration file, keep it free of real credentials.

---

## 🧰 Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| HTTP server | Python `http.server` + `ThreadingMixIn` |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| AI providers | Gemini, Groq, OpenRouter, OpenAI, Anthropic |
| Local inference | Ollama / LM Studio / compatible endpoint |
| Document extraction | PyMuPDF, pypdf, python-docx, python-pptx, openpyxl, Pillow |
| PDF generation | ReportLab |
| Word generation | python-docx |
| Presentation generation | python-pptx / OpenXML fallback |
| Spreadsheet generation | openpyxl |
| Credential/security libraries | cryptography, keyring |

---

## 🗺️ Roadmap

Potential future improvements:

- Production-grade authentication and per-user authorization.
- Persistent cloud database and object storage for hosted use.
- Automated unit and integration test coverage.
- More efficient semantic embeddings and retrieval.
- Faster startup and response-latency profiling.
- Better model capability and provider-health monitoring.
- Improved observability and deployment diagnostics.
- More provider integrations and model options.

---

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Test the affected feature.
5. Do not include API keys, personal documents, local databases, or generated private files.
6. Submit a pull request describing the change and test results.

---


<div align="center">

## 📚 DocPilot AI

**Understand Documents · Learn Smarter · Remember Longer**

Made for focused learning, document intelligence, and AI-powered productivity.

[▶️ Tutorial Video ](https://youtu.be/H4GxjIoe66I) · [🌐 Live Application ](https://docpilot-ai-1.onrender.com/)

</div>
