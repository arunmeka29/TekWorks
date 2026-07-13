"""Streamlit entrypoint for the Day-31 many-to-many RNN app."""

from pathlib import Path
import runpy


APP_FILE = Path(__file__).with_name("manytomany.py")
runpy.run_path(str(APP_FILE), run_name="__main__")
