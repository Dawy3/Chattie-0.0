# Chattie <img width="50" height="50" alt="robot" src="https://github.com/user-attachments/assets/da704aa4-86a0-48ca-b3f7-a2efc5c3a434" />


Chattie is an AI chat assistant that answers questions based on your own documents. Upload files (PDF, Word, TXT, etc.), and Chattie will read, understand, and use them to give you accurate answers.

## What can Chattie do?

- **Answer questions from your documents** — Upload a file and ask anything about its content.
- **Remember your conversation** — Ask follow-up questions naturally.
- **Work fast with caching** — Similar questions get instant answers.
- **Handle multiple documents** — Upload as many files as you need.

## Supported file types

PDF, Word (.docx, .doc), Text (.txt), Markdown (.md), CSV, Excel (.xlsx), HTML

---

## How to run Chattie

You have two options: **Docker (easiest)** or **manual setup**.

---

### Option 1: Using Docker (recommended)

This is the easiest way — you don't need to install Python or Node.js separately.

**Step 1: Install Docker**

- Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Open Docker Desktop and make sure it's running (you'll see a green icon in your taskbar)

**Step 2: Set up your settings**

- In the project folder, find the file called `.env.example`
- Make a copy of it and rename the copy to `.env`
- Open `.env` with any text editor (Notepad works fine)
- Fill in your API keys:
  ```
  OPENAI_API_KEY=your-openai-key-here
  QDRANT_URL=your-qdrant-url-here
  QDRANT_API_KEY=your-qdrant-key-here
  ```
- Save and close the file

**Step 3: Start Chattie**

- Open a terminal (Command Prompt, PowerShell, or Terminal)
- Navigate to the project folder:
  ```
  cd path\to\the\project\folder
  ```
- Run this command:
  ```
  docker compose up
  ```
- Wait until you see messages saying the services are ready (this may take a few minutes the first time)

**Step 4: Open Chattie**

- Open your browser and go to: **http://localhost:3000**
- Click the chat bubble in the bottom-right corner
- You're ready to go!

**To stop Chattie:**
- Press `Ctrl + C` in the terminal, or run:
  ```
  docker compose down
  ```

---

### Option 2: Manual setup (without Docker)

Use this if you prefer not to use Docker.

**What you need to install first:**

- [Python 3.11+](https://www.python.org/downloads/) — during installation, check "Add Python to PATH"
- [Node.js 18+](https://nodejs.org/) — download the LTS version

**Step 1: Set up your settings**

Same as Docker Step 2 above — copy `.env.example` to `.env` and fill in your API keys.

**Step 2: Start the backend**

Open a terminal and run:
```
cd path\to\the\project\folder\backend
pip install -r requirements.txt
python api/main.py
```

You should see a message saying the server is running on `http://localhost:8000`. **Keep this terminal open.**

**Step 3: Start the frontend**

Open a **second** terminal and run:
```
cd path\to\the\project\folder\frontend
npm install
npm run dev
```

You should see a message with a local URL. **Keep this terminal open too.**

**Step 4: Open Chattie**

- Open your browser and go to: **http://localhost:3000**
- Click the chat bubble in the bottom-right corner
- You're ready to go!

**To stop Chattie:**
- Press `Ctrl + C` in both terminals.

---

## How to use Chattie

1. Open **http://localhost:3000** in your browser
2. Click the chat bubble in the bottom-right corner
3. Upload a document at **http://localhost:8000/docs** (this opens an interactive API page — click on `/api/v1/documents/upload`, then "Try it out")
4. Go back to the chat and start asking questions about your document

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Cannot connect" or blank page | Make sure both backend and frontend are running |
| Docker says "port already in use" | Another app is using port 3000 or 8000. Close it and try again |
| "OPENAI_API_KEY not set" error | Make sure you created the `.env` file and added your key |
| Upload fails | Check that your file is one of the supported types and under 50 MB |
