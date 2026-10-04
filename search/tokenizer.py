"""Tokenizer for BM25 (keyword search). Not used for embeddings.

Rules:
1. A "word" is a run of letters, digits or underscores. Everything else
   (spaces, punctuation, brackets, dots) is a separator.
   Underscore is kept inside words so "n_estimators" stays one word.
2. Each word is added lowercased, in full: "ValueError" -> "valueerror",
   "n_estimators" -> "n_estimators". An exact identifier in the query then
   matches the exact identifier in the text, and because such tokens are
   rare their BM25 IDF weight is high.
3. If the word is a compound (snake_case, CamelCase, or letters+digits) we
   ALSO add its parts: "ValueError" -> "value", "error";
   "n_estimators" -> "n", "estimators". So a query written in plain words
   ("value error") still finds the identifier.

Trade-off: a compound word contributes extra tokens, so it counts a little
more towards term frequency and document length. We accept that for the
better recall. No stemming or stop-word removal, to keep it simple and
predictable; BM25's IDF already gives very common words a low weight.
"""

import re

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
# Splits one word into CamelCase / digit parts:
#   "HTTPServer" -> HTTP, Server   "ValueError" -> Value, Error   "bm25" -> bm, 25
PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in WORD_RE.findall(text):
        whole = word.strip("_").lower()
        if not whole:  # word was only underscores
            continue
        tokens.append(whole)
        parts = [part.lower() for piece in word.split("_") for part in PART_RE.findall(piece)]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens
