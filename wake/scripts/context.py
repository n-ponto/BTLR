"""
Add the parent directory to the path to import other modules.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wake import neuralmodels, listener, parameters, audio_collection, preprocessing