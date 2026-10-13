"""Domain-aware text tokenization for technical documentation and code."""

from __future__ import annotations

import re

TOKEN_REGEX = re.compile(r"@[a-zA-Z0-9_]+|[a-zA-Z0-9_]+")
CAMEL_BOUNDARY = re.compile(r"[a-z][A-Z]")
CAMEL_SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)")


def tokenize(text: str) -> list[str]:
    """Tokenize technical text into lowercased words and domain code symbols.

    Domain-Aware Design:
    1. Preserves TypeScript/NestJS Decorators:
       Tokens starting with '@' (e.g. `@Injectable`, `@Controller`, `@UseGuards`)
       are retained with their '@' prefix because decorators carry distinct architectural
       meaning in NestJS separate from ordinary words.
    2. Preserves and Decomposes CamelCase Identifiers:
       Compounds like `AuthGuard`, `canActivate`, and `forRoot` are kept as unified tokens
       (`authguard`, `canactivate`, `forroot`) and also expanded into constituent subwords
       (`auth`, `guard`, `can`, `activate`, `for`, `root`). This ensures queries matching either
       the specific API identifier or the general concept successfully hit the document.
    3. Punctuation and Noise Filtering:
       Syntactic brackets, quotes, and punctuation are stripped while preserving
       alphanumeric tokens.

    Parameters
    ----------
    text : str
        Input technical document or search query string.

    Returns
    -------
    list[str]
        Ordered list of extracted lowercase tokens.
    """
    raw_tokens = TOKEN_REGEX.findall(text)
    tokens: list[str] = []

    for tok in raw_tokens:
        tok_lower = tok.lower()
        tokens.append(tok_lower)

        # Decompose camelCase and PascalCase into subwords
        if not tok.startswith("@") and CAMEL_BOUNDARY.search(tok):
            parts = CAMEL_SPLIT.findall(tok)
            for part in parts:
                part_lower = part.lower()
                if part_lower != tok_lower and len(part_lower) > 1:
                    tokens.append(part_lower)

    return tokens
