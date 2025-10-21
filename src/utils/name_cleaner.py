import re

def normalize_embedding_name(name: str) -> str:
    """
    Normalize an embedding model name for use as a short identifier.

    Args:
        name (str): The full name or path of the embedding model.

    Returns:
        str: A cleaned, alphanumeric, and capitalized short name.
    """
    # Take the last part if the name is a path (e.g., 'folder/model-name' -> 'model-name')
    short_name = name.split('/')[-1]
    # Remove all non-alphanumeric characters from the name
    short_name = re.sub(r'[^0-9a-zA-Z]', '', short_name)
    # Ensure the name starts with an uppercase letter; if not, prepend 'C'
    if not short_name[0].isalpha():
        short_name = "C" + short_name
    return short_name#.lower()


