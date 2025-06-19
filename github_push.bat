@echo off
cd /d "%~dp0"

echo.
echo ==========================
echo Pushing to GitHub...
echo ==========================
echo.

git add .
git commit -m "Sync changes to GitHub"
git push

echo.
echo ==========================
echo Push complete.
echo ==========================
pause
 