
import tiktoken

def chunk_text(text,chunk_size=400,overlap=40):
    tokenizer=tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        print(i)
        chunk_tokens=tokens[i:i + chunk_size]
        chunk_text_decoded = tokenizer.decode(chunk_tokens) 
        chunk_text_decoded = chunk_text_decoded.replace("\n", " ").strip()
        if chunk_text_decoded:
            chunks.append(chunk_text_decoded)        
    return chunks
