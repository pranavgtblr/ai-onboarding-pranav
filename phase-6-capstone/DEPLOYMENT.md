# Deployment Guide for PG Recommends (100% Free)

This guide walks you through deploying the **PG Recommends** full-stack AI application for free.

Because the FastAPI backend automatically serves the compiled Vite/React frontend as static files (`/` and `/assets`), the entire app runs as a **single containerized service**. This means you only need **one free web service** (no separate frontend host, no CORS issues, no subscription fees).

---

## Option 1: Render.com (Recommended — Simplest Free Hosting)

[Render.com](https://render.com) provides a generous **Free Web Service** tier that supports Docker, persistent HTTP streaming (SSE), and free HTTPS without requiring a credit card.

### Prerequisites
1. A GitHub account with this repository pushed.
2. A free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### Step-by-Step Instructions

1. **Sign Up / Log In**: Go to [render.com](https://render.com) and log in with GitHub.
2. **Create New Web Service**:
   - Click **New +** in the top right → select **Web Service**.
   - Connect your GitHub repository (`ai-onboarding-pranav`).
3. **Configure Service**:
   - **Name**: `pg-recommends` (or your choice)
   - **Region**: Choose the closest region (e.g., Singapore, Frankfurt, Oregon)
   - **Root Directory**: `phase-6-capstone`
   - **Environment / Runtime**: Select **Docker**
   - **DockerfilePath**: `Dockerfile` (or `phase-6-capstone/Dockerfile` if Root Directory is left blank)
   - **Instance Type**: Select **Free**
4. **Environment Variables**:
   Under the **Environment Variables** section, add:
   | Key | Value | Notes |
   |---|---|---|
   | `GEMINI_API_KEY` | `your_actual_gemini_api_key` | Secret key from Google AI Studio |
   | `MODEL_NAME` | `gemini-2.0-flash` | Fast & free-tier friendly |
   | `MODEL_PROVIDER` | `google_genai` | Uses Google GenAI SDK |
   | `DATABASE_URL` | `sqlite+aiosqlite:///pg_recommends.db` | Local SQLite database |
5. **Deploy**:
   - Click **Create Web Service**.
   - Render will build the Docker container (builds Vite frontend first, then installs Python dependencies with `uv`, then starts uvicorn).
   - Once deployment completes, you will get a public HTTPS URL: `https://pg-recommends-xxxx.onrender.com`.

> **Note on Render Free Tier**:
> - Free services automatically spin down after 15 minutes of inactivity. The first request after sleep may take ~30–45 seconds to spin back up (cold start).
> - Since SQLite is stored inside the container, data resets when the container restarts. For a persistent database, you can connect a free PostgreSQL database (e.g. Supabase or Neon.tech) by changing `DATABASE_URL`.

---

## Option 2: Hugging Face Spaces (Best for AI Portfolios & Fast Compute)

[Hugging Face Spaces](https://huggingface.co/spaces) offers **16 GB RAM and 2 vCPUs completely free** for Docker apps.

### Step-by-Step Instructions

1. **Create Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Space Name: `pg-recommends`
   - Space SDK: Select **Docker** → **Blank**.
   - Space Hardware: Free (CPU basic · 2 vCPU · 16 GB).
2. **Add Environment Secrets**:
   - In your Space, go to **Settings** → **Variables and secrets**.
   - Add a Secret:
     - Name: `GEMINI_API_KEY`
     - Value: `your_gemini_api_key`
3. **Push Code**:
   - You can push the files in `phase-6-capstone/` to the Hugging Face Git remote:
   ```bash
   cd phase-6-capstone
   git init
   git remote add space https://huggingface.co/spaces/<your-username>/pg-recommends
   git add .
   git commit -m "Deploy PG Recommends"
   git push --force space main
   ```
   - Hugging Face will automatically build the Dockerfile and launch the app on their global CDN.

---

## Option 3: Koyeb (Fast Free Micro Instances)

[Koyeb](https://www.koyeb.com/) offers a free Hobby tier with 512MB RAM and continuous uptime without aggressive sleep timeouts.

1. Create a free account on [koyeb.com](https://www.koyeb.com/).
2. Create an App → Choose **GitHub** as deployment method.
3. Select your repository, set Work Directory to `phase-6-capstone`.
4. Choose **Dockerfile** builder.
5. Add environment variable `GEMINI_API_KEY`.
6. Deploy!

---

## Testing the Docker Container Locally First

To make sure your Docker container builds and runs properly before deploying to the cloud:

```bash
cd phase-6-capstone

# Build the Docker image
docker build -t pg-recommends .

# Run the container (pass your GEMINI_API_KEY)
docker run -p 8000:8000 -e GEMINI_API_KEY="your_api_key_here" pg-recommends
```

Then visit [http://localhost:8000](http://localhost:8000) in your browser. Both the web UI and streaming chat will be active.
