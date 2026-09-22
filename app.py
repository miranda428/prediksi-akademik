"""Compatibility entrypoint for existing Streamlit deployments.

The primary source code is aplikasi.py. This wrapper lets an existing
Streamlit app configured with app.py continue to start without duplicating
the application code.
"""
import aplikasi  # noqa: F401,E402
