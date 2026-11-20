"""SMILES tokenizer for reaction component sequences.

Uses a regex-based approach to split SMILES into chemically meaningful tokens,
supporting atom types in brackets, two-letter elements (Br, Cl), and special
characters for stereochemistry and bonding.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# Standard SMILES tokenization regex (from Schwaller et al., DeepSMILES, etc.)
_SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|"        # bracketed atoms: [C@@H], [nH], [O-], [NH3+], etc.
    r"Br?|Cl?|"           # two-letter elements
    r"N|O|S|P|F|I|"       # single-letter elements
    r"b|c|n|o|s|p|"       # aromatic atoms (lowercase)
    r"\(|\)|"             # branch open/close
    r"\.|"                # fragment separator
    r"=|-|"               # double bond, single bond (also minus in brackets)
    r"#|"                 # triple bond
    r"\\+|"               # explicit bond direction
    r"\/|"                # explicit bond direction
    r":|"                 # aromatic bond (deprecated but still seen)
    r"~|"                 # any bond
    r"@|"                 # tetrahedral chirality
    r"\?|"                # unknown stereochemistry
    r">|"                 # reaction arrow (for full reaction SMILES)
    r"\*|"                # wildcard
    r"\$|"                # ring opening
    r"\%[0-9]{2}|"        # ring closure >9
    r"[0-9])"             # ring closure digits
)

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"      # separates reaction components
CLS_TOKEN = "<cls>"
MASK_TOKEN = "<mask>"    # masked component placeholder

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, CLS_TOKEN, MASK_TOKEN]


class SMILESTokenizer:
    """Character-level SMILES tokenizer with regex-based token splitting.

    Args:
        max_len: maximum sequence length (longer sequences truncated).
        special_tokens: list of special tokens to prepend to vocab.
    """

    def __init__(self, max_len: int = 256, special_tokens: Optional[List[str]] = None):
        self.max_len = max_len
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self._build_vocab()

    def _build_vocab(self):
        # Start with special tokens
        self.token_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def fit(self, smiles_list: List[str]):
        """Extend vocabulary with all tokens found in a corpus of SMILES strings."""
        tokens_seen = set()
        for s in smiles_list:
            tokens_seen.update(self.tokenize_smiles(s))
        for tok in sorted(tokens_seen):
            if tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok

    def tokenize_smiles(self, smiles: str) -> List[str]:
        """Split a SMILES string into tokens."""
        return _SMILES_REGEX.findall(smiles.strip())

    def encode(self, smiles: str, add_special: bool = True) -> List[int]:
        """Encode a SMILES string to token IDs."""
        tokens = self.tokenize_smiles(smiles)
        if add_special:
            tokens = [BOS_TOKEN] + tokens + [EOS_TOKEN]
        # Truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len - 1] + [EOS_TOKEN]
        return [self.token_to_id.get(t, self.token_to_id[UNK_TOKEN]) for t in tokens]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a SMILES string."""
        tokens = [self.id_to_token.get(i, UNK_TOKEN) for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return "".join(tokens)

    def encode_components(self, ligand: str, additive: str, base: str,
                          aryl_halide: str) -> List[int]:
        """Encode reaction components as a flat sequence with separators.

        Format: [BOS] ligand_tokens [SEP] additive_tokens [SEP] base_tokens [SEP]
                aryl_halide_tokens [EOS]
        """
        tokens = [BOS_TOKEN]
        for comp in [ligand, additive, base, aryl_halide]:
            tokens.extend(self.tokenize_smiles(comp))
            tokens.append(SEP_TOKEN)
        tokens[-1] = EOS_TOKEN  # replace last SEP with EOS

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len - 1] + [EOS_TOKEN]

        return [self.token_to_id.get(t, self.token_to_id[UNK_TOKEN]) for t in tokens]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_idx(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token_to_id[UNK_TOKEN]
