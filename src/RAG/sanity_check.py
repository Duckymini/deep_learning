import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from src.RAG.rag import encode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {device}')
print('Loading SBERT...')
tok = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
mdl = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').eval().to(device)

index = faiss.read_index('index/sbert/vdb_training.faiss')
with open('index/lookup_training.json') as f:
    lookup = json.load(f)

print(f'Index: {index.ntotal:,} vectors\n')

queries = [
    ('HATE',     'jews are now in full control : canadian man goes to jail for posting on white nationalist websites'),
    ('HATE',     'TKD GTKRWN'),
    ('HATE',     'You will be drafted to defend zog'),
    ('HATE',     'Based nords commit tkd'),
    ('NOT HATE', 'can they breath fire and have scaly skin ?'),
    ('NOT HATE', 'Charges in deadly wreck that killed 5 motorcyclists .'),
]

THRESHOLD = 0.4

for label, tweet in queries:
    vec = encode([tweet], mdl, tok, batch_size=1, use_mean_pool=True)
    faiss.normalize_L2(vec)
    scores, ids = index.search(vec, 5)
    print(f'[{label}] {tweet}')
    for i in range(5):
        if ids[0][i] != -1 and scores[0][i] >= THRESHOLD:
            print(f'  {scores[0][i]:.3f}  {lookup[str(ids[0][i])][:100]}')
    print()
