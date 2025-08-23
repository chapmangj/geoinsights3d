@echo off
:: This script automatically activates the virtual environment and starts the GeoInsights 3D application.

:: Set the title of the command prompt window
TITLE GeoInsights 3D Launcher

:: Change the current directory to the directory where this script is located.
:: This ensures that the script can find the 'venv' folder correctly, no matter where it's run from.
cd /d "%~dp0"

echo Searching for the virtual environment...

:: Check if the virtual environment's activation script exists.
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo.
    echo ERROR: Virtual environment not found in the 'venv' folder!
    echo Please make sure you have completed the installation steps and that this .bat file is in the correct project folder.
    echo.
    pause
    exit /b
)

echo Virtual environment found. Activating...
echo.

:: Activate the virtual environment. The 'CALL' command is important because it runs the
:: activation script and then returns control back to this script.
CALL venv\Scripts\activate.bat

echo.
echo Starting GeoInsights 3D...
echo Please wait, the application may take a moment to load in your browser.
echo.

:: Run the application.
geoinsights-3d

echo.
echo Application has been closed.
pause