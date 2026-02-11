# Chattie <img width="50" height="50" alt="robot_transparent" src="https://github.com/user-attachments/assets/d785b3b9-fe97-4c5e-b9ac-69955b078272" />
<img width="1857" height="838" alt="Screenshot 2026-02-11 084019" src="https://github.com/user-attachments/assets/d41298f2-c12a-454f-ab60-6b168e0c8681" />



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

## How to use Chattie

1. Open **http://localhost:3000** in your browser
2. Click the chat bubble in the bottom-right corner
3. Upload a document at **http://localhost:8000/docs** (this opens an interactive API page — click on `/api/v1/documents/upload`, then "Try it out")
4. Go back to the chat and start asking questions about your document

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Cannot connect" or blank page | Make sure Docker Desktop is running and you ran `docker compose up` |
| Docker says "port already in use" | Another app is using port 3000 or 8000. Close it and try again |
| "OPENAI_API_KEY not set" error | Make sure you created the `.env` file and added your key |
| Upload fails | Check that your file is one of the supported types and under 50 MB |
