"""
This file is used to add the parent directory to the path so that the modules
can be imported from the scripts directory.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wake import neuralmodels, listener, parameters, audio_collection, preprocessing