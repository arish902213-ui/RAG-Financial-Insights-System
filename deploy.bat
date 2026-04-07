@echo off
REM Streamlit Cloud Deployment Helper for Windows

echo.
echo ========================================
echo RAG Financial Insights - Deployment Helper
echo ========================================
echo.

REM Check if Git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Git is not installed!
    echo.
    echo Please download and install Git from: https://git-scm.com/download/win
    echo Then run this script again.
    echo.
    pause
    exit /b 1
)

echo ✓ Git is installed

REM Initialize git repository
if not exist .git (
    echo.
    echo Initializing Git repository...
    git init
    git add .
    git commit -m "Initial commit: RAG-Based Financial Insights System"
    echo ✓ Git repository initialized
) else (
    echo ✓ Git repository already initialized
)

echo.
echo ========================================
echo Next Steps:
echo ========================================
echo.
echo 1. Create a GitHub account at https://github.com
echo.
echo 2. Create a new repository:
echo    - Name: RAG-Based-Knowledge-Financial-Insights-System
echo    - Make it PUBLIC
echo.
echo 3. After creating the repo, run these commands:
echo    (Replace YOUR_USERNAME with your GitHub username)
echo.
echo    git branch -M main
echo    git remote add origin https://github.com/YOUR_USERNAME/RAG-Based-Knowledge-Financial-Insights-System.git
echo    git push -u origin main
echo.
echo 4. Deploy on Streamlit Cloud:
echo    - Go to https://share.streamlit.io
echo    - Sign in with GitHub
echo    - Click "New app"
echo    - Select your GitHub repo and main file as app.py
echo    - Click "Deploy!"
echo.
echo 5. (Optional) Add OpenAI API key:
echo    - In Streamlit Cloud dashboard, go to app settings
echo    - Add secret: OPENAI_API_KEY = "your-key-here"
echo.
echo ========================================
echo.
echo For detailed instructions, see: DEPLOYMENT.md
echo.
pause
