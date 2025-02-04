"""Tests for Gradio interface functionality"""
import pytest
from voice_assistant.ui.gradio_interface import create_interface
import gradio as gr

def test_interface_creation():
    """Test interface initialization"""
    interface = create_interface()
    assert isinstance(interface, gr.Blocks)

def test_safe_start_recording():
    """Test recording start handler"""
    interface = create_interface()
    
    # Get the start recording function
    start_fn = interface.fns[0]  # First function should be safe_start_recording
    
    # Test the function
    result = start_fn()
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(x, (gr.update, str)) for x in result)

def test_safe_stop_audio():
    """Test audio stop handler"""
    interface = create_interface()
    
    # Get the stop audio function
    stop_fn = interface.fns[2]  # Third function should be safe_stop_audio
    
    # Test the function
    result = stop_fn()
    assert isinstance(result, str) 