import hashlib

def generate_id_from_strings(*strings):
    # Concatenate all strings
    combined = ''.join(str(s) for s in strings)
    # Generate SHA-256 hash
    return hashlib.sha256(combined.encode()).hexdigest()