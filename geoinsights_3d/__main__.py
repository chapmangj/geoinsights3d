# In geoinsights_3d/__main__.py

import os
import sys
import subprocess
import importlib.resources

def main():
    """
    The entry point for the command-line script.
    This launches the Streamlit app in a robust way.
    """
    # Find the absolute path to the app.py file within the installed package
    try:
        app_path_obj = importlib.resources.files('geoinsights_3d').joinpath('app.py')
        app_path = str(app_path_obj)
    except AttributeError:
        # Fallback for older importlib_resources or edge cases
        with importlib.resources.path('geoinsights_3d', 'app.py') as p:
            app_path = str(p)

    # Get the user's current working directory. 
    user_cwd = os.getcwd()

    # Construct the command to run Streamlit

    command = [sys.executable, "-m", "streamlit", "run", app_path]

    # Run the Streamlit command as a subprocess, 
    # set the current working directory (cwd) to the user's directory.

    try:
        subprocess.run(command, cwd=user_cwd, check=True)
    except FileNotFoundError:
        print(
            "Error: 'streamlit' command not found.",
            "Please ensure Streamlit is installed correctly in your environment.",
            file=sys.stderr
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()