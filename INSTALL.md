# Installation Guide

This guide provides detailed installation instructions for all platforms.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Linux Installation](#linux-installation)
- [Docker Installation (Optional)](#docker-installation-optional)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Prerequisites

Before installing MOTO Math Variant, ensure you have:

### Required Software

1. **Python 3.8 or higher**
   - Windows: https://www.python.org/downloads/
   - macOS: Use Homebrew `brew install python3` or download from python.org
   - Linux: Usually pre-installed, or `sudo apt install python3 python3-pip`
   - ⚠️ **IMPORTANT**: On Windows, check "Add Python to PATH" during installation

2. **Node.js 16 or higher**
   - All platforms: https://nodejs.org/ (download LTS version)
   - Or use nvm (Node Version Manager):
     - macOS/Linux: https://github.com/nvm-sh/nvm
     - Windows: https://github.com/coreybutler/nvm-windows

3. **Git** (for cloning the repository)
   - Windows: https://git-scm.com/download/win
   - macOS: Comes with Xcode Command Line Tools or `brew install git`
   - Linux: `sudo apt install git` (Ubuntu/Debian)

### Recommended Software

4. **LM Studio** (required for the nomic embedding agent (for now) optional for local submitter/validator models)
   - All platforms: https://lmstudio.ai/
   - Alternative: Use OpenRouter API (cloud-based, no local installation)

### System Requirements

- **RAM**: 16GB minimum, 32GB+ recommended (for local models)
- **Storage**: 10GB free space minimum, 50GB+ recommended (for multiple models)
- **GPU**: Optional but recommended - NVIDIA GPU with 8GB+ VRAM for faster inference

---

## Windows Installation

### Method 1: One-Click Launcher (Easiest)

1. **Install Prerequisites**:
   - Download and install [Python 3.8+](https://www.python.org/downloads/)
     - ⚠️ **Check "Add Python to PATH"** during installation
   - Download and install [Node.js 16+](https://nodejs.org/)
   - Download and install [LM Studio](https://lmstudio.ai/) (optional)

2. **Download MOTO**:
   ```bash
   git clone https://github.com/Intrafere/MOTO
   cd moto-math-variant
   ```

3. **Start LM Studio** (if using local models):
   - Open LM Studio
   - Download and load a model (e.g., DeepSeek R1, Llama 3.1, Qwen 2.5)
   - **REQUIRED**: Download and load `nomic-ai/nomic-embed-text-v1.5`
   - Go to "Local Server" tab
   - Click "Start Server"

4. **Launch MOTO**:
   - Double-click `launch.bat`
   - The launcher will automatically:
     - Check Python and Node.js
     - Install all dependencies
     - Create necessary directories
     - Start the system
     - Open browser to UI

### Method 2: Manual Installation (Windows)

1. **Install Prerequisites** (same as Method 1)

2. **Clone Repository**:
   ```bash
   git clone https://github.com/Intrafere/MOTO
   cd moto-math-variant
   ```

3. **Create Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Frontend Dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

6. **Create Required Directories**:
   ```bash
   mkdir backend\data\user_uploads
   mkdir backend\logs
   ```

7. **Start Backend** (in one terminal):
   ```bash
   python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
   ```

8. **Start Frontend** (in another terminal):
   ```bash
   cd frontend
   npm run dev
   ```

9. **Access UI**: Open browser to `http://localhost:5173`

---

## macOS Installation

### Method 1: Using Homebrew

1. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Prerequisites**:
   ```bash
   brew install python3 node git
   ```

3. **Download LM Studio**:
   - Download from https://lmstudio.ai/
   - Or use OpenRouter API (no local installation)

4. **Clone Repository**:
   ```bash
   git clone https://github.com/Intrafere/MOTO
   cd moto-math-variant
   ```

5. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

6. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   cd frontend
   npm install
   cd ..
   ```

7. **Create Required Directories**:
   ```bash
   mkdir -p backend/data/user_uploads
   mkdir -p backend/logs
   ```

8. **Start Services**:
   
   Terminal 1 (Backend):
   ```bash
   source venv/bin/activate
   python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
   ```
   
   Terminal 2 (Frontend):
   ```bash
   cd frontend
   npm run dev
   ```

9. **Access UI**: Open browser to `http://localhost:5173`

### Method 2: Using PowerShell Launcher

macOS supports the PowerShell launcher:

```bash
# Install PowerShell (if not installed)
brew install --cask powershell

# Run launcher
pwsh launch.ps1
```

---

## Linux Installation

### Ubuntu/Debian

1. **Install Prerequisites**:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv nodejs npm git
   ```

2. **Clone Repository**:
   ```bash
   git clone https://github.com/Intrafere/MOTO
   cd moto-math-variant
   ```

3. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   cd frontend
   npm install
   cd ..
   ```

5. **Create Required Directories**:
   ```bash
   mkdir -p backend/data/user_uploads
   mkdir -p backend/logs
   ```

6. **Start Services**:
   
   Terminal 1 (Backend):
   ```bash
   source venv/bin/activate
   python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
   ```
   
   Terminal 2 (Frontend):
   ```bash
   cd frontend
   npm run dev
   ```

7. **Access UI**: Open browser to `http://localhost:5173`

### Fedora/CentOS/RHEL

1. **Install Prerequisites**:
   ```bash
   sudo dnf install -y python3 python3-pip nodejs npm git
   ```

2. Follow steps 2-7 from Ubuntu/Debian instructions above.

### Arch Linux

1. **Install Prerequisites**:
   ```bash
   sudo pacman -S python python-pip nodejs npm git
   ```

2. Follow steps 2-7 from Ubuntu/Debian instructions above.

---

## Docker Installation (Optional)

Coming soon - Docker support is planned for future releases.

---

## Troubleshooting

### Python Issues

**"python: command not found"**
- Try `python3` instead of `python`
- Add Python to PATH (see platform-specific instructions)

**"pip: command not found"**
- Try `python -m pip` or `python3 -m pip`
- Install pip: `python -m ensurepip --upgrade`

**"Permission denied" errors**
- Don't use `sudo pip` - use virtual environment instead
- Or install with `--user` flag: `pip install --user -r requirements.txt`

### Node.js Issues

**"node: command not found"**
- Verify Node.js installation: `node --version`
- Reinstall Node.js from nodejs.org

**"npm: command not found"**
- npm should come with Node.js
- Reinstall Node.js

**"EACCES" permission errors**
- Fix npm permissions: https://docs.npmjs.com/resolving-eacces-permissions-errors-when-installing-packages-globally

### Port Conflicts

**"Port 8000 already in use"**
- Find and kill process:
  - Windows: `netstat -ano | findstr :8000` then `taskkill /F /PID <PID>`
  - macOS/Linux: `lsof -ti:8000 | xargs kill -9`

**"Port 5173 already in use"**
- Same as above but for port 5173

### LM Studio Issues

**"Cannot connect to LM Studio"**
- Ensure LM Studio is running
- Start the local server in LM Studio (port 1234)
- Load at least one model
- Load embedding model: `nomic-ai/nomic-embed-text-v1.5`

**"Model not found"**
- Download models in LM Studio
- Wait for model to fully load before starting MOTO

### Memory Issues

**"Out of memory" errors**
- Use smaller models
- Reduce context window size in settings
- Close other applications
- Upgrade RAM if possible

---

## Verification

After installation, verify everything works:

### 1. Check Prerequisites

```bash
# Python
python --version  # or python3 --version
# Should show: Python 3.8.x or higher

# Node.js
node --version
# Should show: v16.x.x or higher

# npm
npm --version
# Should show: 8.x.x or higher

# Git
git --version
# Should show: git version 2.x.x
```

### 2. Check Services

- **Backend**: http://localhost:8000/docs (should show API documentation)
- **Frontend**: http://localhost:5173 (should show UI)
- **LM Studio**: http://127.0.0.1:1234/v1/models (should list models)

### 3. Run Test

1. Go to Aggregator Interface
2. Enter test prompt: "What is 2 + 2?"
3. Click "Start Aggregator"
4. Should see submissions being generated and validated

---

## Next Steps

- Read [QUICKSTART.md](QUICKSTART.md) for usage guide
- Read [README.md](README.md) for system overview
- Check `.cursor/rules/` for detailed design specifications
- Join discussions for support

---

## Support

If you encounter issues:
1. Check troubleshooting section above
2. Search existing issues on GitHub
3. Open a new issue with detailed error logs
4. Include system info (OS, Python version, Node version, etc.)

---

**Installation complete! Ready to start your autonomous research journey.** 🚀

