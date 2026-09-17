from ollama import embed

response = embed(model='nomic-embed-text', input='SUID binaries can be abused for privilege escalation')
print(response['embeddings'])
# test
# test
