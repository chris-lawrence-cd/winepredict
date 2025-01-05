import pytest
from winepredict import info

def test_info_contents():
    """Test that the info module contains expected attributes."""
    assert hasattr(info, 'version'), "Info should have a 'version' attribute"
    assert hasattr(info, 'author'), "Info should have an 'author' attribute"
    assert isinstance(info.version, str), "Version should be a string"
    assert isinstance(info.author, str), "Author should be a string"

    # Example checks, adjust according to actual contents of info module
    assert info.version == '1.0', "Version should be '1.0'"
    assert info.author == 'Chris Lawrence', "Author should be 'Chris Lawrence'"
