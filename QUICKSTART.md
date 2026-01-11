# Quick Start Guide - ASI Aggregator System

## 🚀 ONE-CLICK LAUNCHER (RECOMMENDED)

**The easiest way to start the system:**

### Windows Users
1. Make sure you have Python 3.8+, Node.js 16+, and LM Studio (optional) installed
2. Start LM Studio and load a model
3. **Double-click `launch.bat`** in the project root
4. Follow the prompts - the launcher will:
   - Check all dependencies
   - Install missing packages automatically
   - Start both backend and frontend
   - Open the UI in your browser

---

## Manual Setup (Alternative Method)

If the one-click launcher doesn't work, follow these manual steps:

### Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.10 or higher
- [ ] Node.js 18 or higher
- [ ] LM Studio installed and running

## (OPTIONAL) Step 1: Install LM Studio and Load Models

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Launch LM Studio
3. Load at least one model for generation (e.g., Llama-3.2, DeepSeek, Mistral)
4. Load the embedding model: `nomic-ai/nomic-embed-text-v1.5`
5. Go to "Local Server" tab and click "Start Server"
6. Verify server is running on `http://127.0.0.1:1234`

## Step 2: Install Dependencies

```bash
# Navigate to project directory
cd moto-math-variant

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
npm run install:frontend
```

## Step 3: Create Required Directories

```bash
# Create data directories
mkdir -p backend/data/user_uploads
mkdir -p backend/data/chroma_db
mkdir -p backend/logs
```

## Step 4: Start the Backend

Open a terminal and run:

```bash
# Option 1: Using npm script
npm run dev:backend

# Option 2: Direct Python command
python -m backend.api.main
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 5: Start the Frontend

Open a **second terminal** and run:

```bash
# Option 1: Using npm script
npm run dev:frontend

# Option 2: Direct command
cd frontend && npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

## Step 6: Access the Application

1. Open your browser to `http://localhost:5173`
2. You should see the ASI Aggregator System interface

## Step 7: Configure Settings

1. Click on the "Aggregator Settings" tab
2. Select your submitter model from the dropdown (should auto-populate from LM Studio)
3. Select your validator model
4. Leave context size at 8192 (default) or adjust based on your model

## Step 8: Run Your First Aggregation

1. Go to "Aggregator Interface" tab
2. Enter a research prompt, for example:
   ```
   Explain the relationship between quantum decoherence and the emergence of classical behavior in macroscopic systems.
   ```
3. (Optional) Upload reference files using the file input
4. Click "Start Aggregator"

## Step 9: Monitor Progress

1. **Aggregator Logs Tab**: Watch real-time submissions, acceptances, and rejections
2. **Live Results Tab**: See accepted submissions as they're validated
3. Monitor metrics:
   - Total submissions
   - Acceptance rate
   - Queue size
   - Per-submitter statistics

## Step 10: Save Results

1. Go to "Live Results" tab
2. Click "Save Results to File"
3. Results saved to `backend/data/aggregator_results.txt`

## Troubleshooting

### "Failed to connect to LM Studio"
**Solution**: Ensure LM Studio local server is running on port 1234

### "No models found"
**Solution**: 
- Load at least one model in LM Studio
- Click "Refresh Models" in Settings tab
- Restart LM Studio server

### Backend won't start
**Solution**:
- Check Python version: `python --version` (need 3.10+)
- Install dependencies: `pip install -r requirements.txt`
- Check port 8000 is not in use

### Frontend won't start
**Solution**:
- Check Node version: `node --version` (need 18+)
- Install dependencies: `cd frontend && npm install`
- Check port 5173 is not in use

### High rejection rate
**Solution**:
- Check that models are generating valid JSON
- Review validator reasoning in Logs tab
- Ensure prompt is clear and specific

### System running slow
**Solution**:
- Use faster/smaller models in Settings
- Reduce context window size
- Close other resource-intensive applications

## Next Steps

- Review `IMPLEMENTATION_SUMMARY.md` for architecture details
- Read `backend/compiler/README.md` for compiler documentation
- Check summary files for recent updates

## Support

If you encounter issues:
1. Check the backend logs in terminal
2. Check the frontend console (F12 in browser)
3. Review `backend/logs/system.log`
4. Ensure all prerequisites are met

## Tips for Best Results

1. **Start Simple**: Use clear, focused prompts for first tests
2. **Monitor Logs**: Watch acceptance/rejection patterns to tune your prompts
3. **Upload Context**: Provide reference files for better results
4. **Save Often**: Use "Save Results" regularly to preserve progress
5. **Experiment**: Try different models and context sizes

---

🎉 **You're all set!** The ASI Aggregator System is now running and ready to process your research prompts.

