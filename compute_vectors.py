from __future__ import annotations

import bz2
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import torch
import ipadic
from fugashi import GenericTagger
from scipy.linalg import norm
from transformers import BertJapaneseTokenizer, BertModel

WORD = "拓殖"
SENTENCES = [
    "すみません、お店は開いていますか？",
    "ご迷惑をおかけしてすみませんでした。",
]
W2V_URL = (
    "https://github.com/singletongue/WikiEntVec/releases/download/20190520/"
    "jawiki.word_vectors.200d.txt.bz2"
)
W2V_PATH = Path("jawiki.word_vectors.200d.txt.bz2")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)


def download_with_retries(url: str, destination: Path, attempts: int = 4) -> None:
    if destination.exists() and destination.stat().st_size > 600_000_000:
        print(f"Using existing {destination} ({destination.stat().st_size:,} bytes)")
        return

    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading Word2Vec model (attempt {attempt}/{attempts})...")
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                written = 0
                with destination.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        written += len(chunk)
                        if total:
                            print(f"  {written / total:6.1%}", end="\r", flush=True)
            print(f"\nDownloaded {destination.stat().st_size:,} bytes")
            return
        except Exception as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            destination.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)


def tokenize_with_ipadic(sentences: Iterable[str]) -> list[list[str]]:
    tagger = GenericTagger(ipadic.MECAB_ARGS)
    return [[word.surface for word in tagger(sentence)] for sentence in sentences]


def collect_word2vec_vectors(
    compressed_path: Path, target_words: set[str]
) -> tuple[dict[str, np.ndarray], str]:
    found: dict[str, np.ndarray] = {}
    with bz2.open(compressed_path, "rt", encoding="utf-8", errors="strict") as source:
        header = next(source).strip()
        print(f"Word2Vec header: {header}")
        for line_no, line in enumerate(source, start=2):
            token, separator, values_text = line.partition(" ")
            if separator and token in target_words:
                vector = np.fromstring(values_text, dtype=np.float32, sep=" ")
                if vector.shape != (200,):
                    raise ValueError(
                        f"Unexpected vector shape for {token!r}: {vector.shape} at line {line_no}"
                    )
                found[token] = vector
                print(f"Found Word2Vec vector: {token!r} ({len(found)}/{len(target_words)})")
                if len(found) == len(target_words):
                    break

    missing = sorted(target_words - set(found))
    if missing:
        raise KeyError(f"Words not found in WikiEntVec: {missing}")
    return found, header


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print("Tokenizing example sentences with fugashi + IPA dictionary...")
    sentence_tokens = tokenize_with_ipadic(SENTENCES)
    for index, tokens in enumerate(sentence_tokens):
        print(f"sentence_{index}: {tokens}")

    target_words = {WORD}
    for tokens in sentence_tokens:
        target_words.update(tokens)

    download_with_retries(W2V_URL, W2V_PATH)
    w2v_vectors, w2v_header = collect_word2vec_vectors(W2V_PATH, target_words)

    word2vec_vector = w2v_vectors[WORD]
    sentence_w2v_vectors = np.stack(
        [
            np.mean(np.stack([w2v_vectors[token] for token in tokens]), axis=0)
            for tokens in sentence_tokens
        ]
    ).astype(np.float32)
    word2vec_similarity = cosine(sentence_w2v_vectors[0], sentence_w2v_vectors[1])

    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    print(f"Loading BERT tokenizer/model: {model_name}")
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)
    bert_model.eval()

    word_inputs = tokenizer(WORD, return_tensors="pt")
    with torch.no_grad():
        word_outputs = bert_model(**word_inputs)
    last_hidden_state = word_outputs.last_hidden_state
    bert_word_vector = last_hidden_state[0][1].detach().cpu().numpy().astype(np.float32)
    word_tokens_with_special = tokenizer.convert_ids_to_tokens(word_inputs["input_ids"][0])

    sentence_inputs = tokenizer(
        SENTENCES,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        sentence_outputs = bert_model(**sentence_inputs)
    sentence_hidden_states = sentence_outputs.last_hidden_state
    attention_mask = sentence_inputs.attention_mask.unsqueeze(-1)
    valid_token_num = attention_mask.sum(1)
    sentence_bert_vectors = (
        (sentence_hidden_states * attention_mask).sum(1) / valid_token_num
    ).detach().cpu().numpy().astype(np.float32)
    bert_similarity = cosine(sentence_bert_vectors[0], sentence_bert_vectors[1])

    word2vec_payload = {
        "word": WORD,
        "model": "WikiEntVec 20190520 jawiki.word_vectors.200d.txt",
        "source_url": W2V_URL,
        "header": w2v_header,
        "shape": list(word2vec_vector.shape),
        "dtype": str(word2vec_vector.dtype),
        "vector": [float(value) for value in word2vec_vector],
    }
    bert_payload = {
        "word": WORD,
        "model": model_name,
        "tokens": word_tokens_with_special,
        "input_ids": [int(value) for value in word_inputs["input_ids"][0]],
        "last_hidden_state_shape": list(last_hidden_state.shape),
        "shape": list(bert_word_vector.shape),
        "dtype": str(bert_word_vector.dtype),
        "vector": [float(value) for value in bert_word_vector],
    }
    similarity_payload = {
        "sentences": SENTENCES,
        "word2vec_tokens": sentence_tokens,
        "bert_input_tokens": [
            tokenizer.convert_ids_to_tokens(row[: int(mask.sum())])
            for row, mask in zip(sentence_inputs["input_ids"], sentence_inputs["attention_mask"])
        ],
        "bert_cosine_similarity": bert_similarity,
        "word2vec_cosine_similarity": word2vec_similarity,
        "lecture_reference_values": {
            "bert": 0.78672804,
            "word2vec": 0.8954789,
        },
    }

    save_json(RESULT_DIR / "word2vec_takushoku.json", word2vec_payload)
    save_json(RESULT_DIR / "bert_takushoku.json", bert_payload)
    save_json(RESULT_DIR / "sentence_similarities.json", similarity_payload)

    np.set_printoptions(precision=8, threshold=np.inf, linewidth=110)
    (RESULT_DIR / "word2vec_output.txt").write_text(
        "word = '拓殖'\n"
        f"Word2Vecで作成した単語ベクトルのshape： {word2vec_vector.shape}\n"
        f"Word2Vecで作成した単語ベクトル： {word2vec_vector!r}\n",
        encoding="utf-8",
    )
    torch.set_printoptions(precision=4, threshold=sys.maxsize, linewidth=110, sci_mode=True)
    bert_tensor = torch.from_numpy(bert_word_vector)
    (RESULT_DIR / "bert_output.txt").write_text(
        "word = '拓殖'\n"
        f"tokens： {word_tokens_with_special}\n"
        f"input_ids： {word_inputs['input_ids']}\n"
        f"最終層のテンソルのshape： {last_hidden_state.shape}\n"
        f"BERTで作成した単語ベクトルのshape： {bert_tensor.shape}\n"
        f"BERTで作成した単語ベクトル： {bert_tensor}\n",
        encoding="utf-8",
    )
    (RESULT_DIR / "similarity_output.txt").write_text(
        f"BERTで計算したコサイン類似度： {bert_similarity:.8f}\n"
        f"Word2Vecで計算したコサイン類似度： {word2vec_similarity:.7f}\n",
        encoding="utf-8",
    )

    print("\n=== Results ===")
    print(f"Word2Vec vector shape: {word2vec_vector.shape}")
    print(f"BERT tokens: {word_tokens_with_special}")
    print(f"BERT input IDs: {word_inputs['input_ids'].tolist()}")
    print(f"BERT vector shape: {bert_word_vector.shape}")
    print(f"BERT cosine similarity: {bert_similarity:.8f}")
    print(f"Word2Vec cosine similarity: {word2vec_similarity:.7f}")


if __name__ == "__main__":
    main()
