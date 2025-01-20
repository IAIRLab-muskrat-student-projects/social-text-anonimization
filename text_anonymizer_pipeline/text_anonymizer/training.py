import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import chain
import re
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# CONSTANTS
BATCH_SIZE = 64
MODEL_NAME = 'distilbert-base-multilingual-cased'
CHECKPOINT_PATH = "checkpoints"
MAX_SEQ_LEN = 128

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

def debug_lengths(embeddings, input_ids):
    print("=== Debugging Embeddings ===")
    for i, emb in enumerate(embeddings[:5]):  # Проверяем первые 5 записей
        print(f"Embedding {i}: Shape = {emb.size()}")

    print("\n=== Debugging Input IDs ===")
    for i, ids in enumerate(input_ids[:5]):
        print(f"Input IDs {i}: Shape = {ids.size()}")

# === Tokenizer and Vocabulary Setup ===
def get_words(doc):
    """Tokenize a document and split into words."""
    doc = tokenizer.decode(tokenizer.encode(doc, max_length=MAX_SEQ_LEN, padding='max_length', truncation=True))
    doc = re.sub(r'([\.,\'’\"\-!\?\(\)])', r' \1 ', doc)
    doc = re.sub('\s', ' ', doc)
    return doc.split()


# Build vocabulary from train and test datasets
def build_vocab(train_posts, test_posts):
    unique_tokens = set(chain.from_iterable([
        *train_posts.text.apply(get_words).tolist(),
        *test_posts.text.apply(get_words).tolist()
    ]))
    token2idx = {token: idx for idx, token in enumerate(unique_tokens)}
    idx2token = {idx: token for idx, token in enumerate(unique_tokens)}
    return token2idx, idx2token, len(unique_tokens)


# === Embeddings and Dataset Preparation ===
def get_hidden_states(encoded, token_ids_words, model, layers):
    """Extract hidden states for token embeddings."""
    with torch.no_grad():
        output = model(**encoded)
    states = output.hidden_states
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    res, labels_count = [], []
    for idx, (outp, label) in enumerate(zip(output, token_ids_words)):
        if label is None or token_ids_words[idx - 1] is None or token_ids_words[idx - 1] != token_ids_words[idx]:
            res.append(outp)
            labels_count.append(1)
        else:
            res[-1] += outp
            labels_count[-1] += 1
    res = torch.vstack(res)
    res = res / torch.tensor(labels_count).float().unsqueeze(1)
    return res


def get_word_vectors(sent, tokenizer, model, layers, token2idx):
    """Extract word embeddings and token IDs."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt", max_length=MAX_SEQ_LEN, padding='max_length', truncation=True)
    input_ids = list(map(lambda x: token2idx[x], get_words(sent)))
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    return get_hidden_states(encoded, encoded.word_ids(), model, layers).cpu(), input_ids


def prepare_embeddings(posts, tokenizer, model, token2idx):
    """Generate embeddings and token IDs for training."""
    embeddings, input_ids = [], []
    token_embeddings = torch.zeros((len(token2idx), 768))
    token_count = torch.zeros((len(token2idx),))

    for idx, doc in enumerate(posts.text):
        try:
            embedding, ids = get_word_vectors(doc, tokenizer, model, [-4, -3, -2, -1], token2idx)

            # Обрезка или дополнение embeddings
            if embedding.size(0) > MAX_SEQ_LEN:
                embedding = embedding[:MAX_SEQ_LEN]
            elif embedding.size(0) < MAX_SEQ_LEN:
                pad_len = MAX_SEQ_LEN - embedding.size(0)
                embedding = torch.cat((embedding, torch.zeros(pad_len, embedding.size(1))), dim=0)

            # Обрезка или дополнение input_ids
            if ids.size(1) > MAX_SEQ_LEN:
                ids = ids[:, :MAX_SEQ_LEN]
            elif ids.size(1) < MAX_SEQ_LEN:
                pad_len = MAX_SEQ_LEN - ids.size(1)
                ids = torch.cat((ids, torch.full((1, pad_len), PAD_TOKEN_ID)), dim=1)

            embeddings.append(embedding)
            input_ids.append(ids)

            token_embeddings.scatter_add_(0, ids[0].unsqueeze(1).expand(embedding.shape), embedding)
            token_count.scatter_add_(0, ids[0], torch.ones_like(ids[0]).float())
        except Exception as e:
            print(f"Error at index {idx}: {e}")

    token_embeddings = torch.nan_to_num(token_embeddings / token_count.unsqueeze(1))
    return embeddings, input_ids, token_embeddings



# === Dataset Definition ===
class NYDataset(Dataset):
    def __init__(self, embeddings, input_ids, pad_token_id):
        # Дополнительно проверяем и обрезаем длины
        max_len = max(seq.size(0) for seq in embeddings)
        embeddings = [torch.cat((seq, torch.zeros(max_len - seq.size(0), seq.size(1))), dim=0) if seq.size(0) < max_len else seq for seq in embeddings]
        input_ids = [torch.cat((seq, torch.full((1, max_len - seq.size(1)), pad_token_id)), dim=1) if seq.size(1) < max_len else seq for seq in input_ids]

        # Применяем pad_sequence
        self.embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0)
        self.input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.input_ids[idx]



# === Loss Functions ===
def recon_loss(inp, targ):
    return F.cross_entropy(inp.view(-1, inp.size(-1)), targ.view(-1))


def embed_loss(inp, targ, token_similarities, k=5):
    loss = 0
    inp = F.log_softmax(inp, dim=-1)
    for i in range(inp.size(0)):
        topk_values, topk_indices = torch.topk(inp[i], k, dim=-1)
        loss += (topk_values * token_similarities[targ[i], topk_indices]).sum()
    return -loss


def total_loss(inp, targ, token_similarities, alpha=1, beta=0.5, k=5):
    return alpha * recon_loss(inp, targ) + beta * embed_loss(inp, targ, token_similarities, k)


# === Model Definition ===
class LitERAE(pl.LightningModule):
    def __init__(self, dataset, vocab_size, token_similarities, hidden_size=512, learning_rate=1e-3, stage="fit"):
        super().__init__()
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.token_similarities = token_similarities
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.stage = stage

        self.gru_1 = nn.GRU(768, hidden_size, bidirectional=True, batch_first=True)
        self.linear_1 = nn.Linear(hidden_size * 2, 768)
        self.gru_2 = nn.GRU(768, hidden_size, bidirectional=True, batch_first=True)
        self.linear_2 = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x, hidden = self.gru_1(x)
        x = F.relu(self.linear_1(x))
        x, _ = self.gru_2(x, hidden)
        return self.linear_2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.stage == "fit":
            loss = recon_loss(logits, y)
        else:
            loss = total_loss(logits, y, self.token_similarities)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# === Training Pipeline ===
def train_model(train_posts, test_posts, checkpoint_path):
    # Build vocabulary and prepare embeddings
    token2idx, idx2token, vocab_size = build_vocab(train_posts, test_posts)
    train_embeddings, train_input_ids, token_embeddings = prepare_embeddings(train_posts, tokenizer, bert, token2idx)
    test_embeddings, test_input_ids, _ = prepare_embeddings(test_posts, tokenizer, bert, token2idx)

    # Calculate token similarities
    token_similarities = cosine_similarity(token_embeddings, token_embeddings)
    token_similarities = torch.tensor(token_similarities).clamp(max=0.85)

    debug_lengths(train_embeddings, train_input_ids)


    # Create datasets
    train_dataset = NYDataset(train_embeddings, train_input_ids, token2idx['[PAD]'])
    test_dataset = NYDataset(test_embeddings, test_input_ids, token2idx['[PAD]'])


    # Initialize model and trainer
    model = LitERAE(train_dataset, vocab_size, token_similarities)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path, save_top_k=2, monitor="val_loss")

    trainer = Trainer(max_epochs=20, callbacks=[checkpoint_callback])
    trainer.fit(model)

    # Second stage of training
    model.stage = "fit_2"
    trainer.fit(model)

    return trainer

