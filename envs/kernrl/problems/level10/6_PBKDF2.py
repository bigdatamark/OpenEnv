# [MODULE DOC]
# kernrl problem (level 10): PBKDF2-style key derivation (default 1000
# iterations, 32-byte output) built on a toy byte-mixing "HMAC" (not real
# HMAC-SHA256), so outputs are only self-consistent.
"""
PBKDF2 Key Derivation

Password-Based Key Derivation Function 2.
Derives cryptographic keys from passwords with salt and iteration count.

Used for secure password storage and key generation.

DK = PBKDF2(Password, Salt, c, dkLen)
where c is iteration count (high for security).

Optimization opportunities:
- Parallel HMAC computation
- Unrolled inner loops
- Shared memory for intermediate hashes
- Multiple derived blocks in parallel
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    # [CLASS DOC]
    # Reference model storing the iteration count and derived key length.
    """
    PBKDF2-HMAC-SHA256 key derivation.

    Simplified implementation for kernel optimization practice.
    """

    def __init__(self, iterations: int = 1000, dk_len: int = 32):
        # [FUNCTION DOC]
        # Stores iterations (default 1000) and dk_len (default 32).
        super(Model, self).__init__()
        self.iterations = iterations
        self.dk_len = dk_len

    def _xor(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Elementwise XOR of two tensors.
        """XOR two byte tensors."""
        return a ^ b

    def _simple_hmac(self, key: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Toy PRF: accumulates key followed by message into 32 byte slots
        # (slot = slot*31 + byte, mod 256), then 4 mixing passes combining each slot
        # with slots +17 and +11. Returns a (32,) int64 tensor.
        """Simplified HMAC (not cryptographically secure - for demo)."""
        # Real HMAC-SHA256 would be: H(key ^ opad || H(key ^ ipad || message))
        # This is a placeholder that produces consistent output
        result = torch.zeros(32, dtype=torch.int64, device=key.device)

        # Mix key and message
        combined = torch.cat([key, message])
        for i in range(len(combined)):
            result[i % 32] = (result[i % 32] * 31 + combined[i]) & 0xFF

        # Additional mixing
        for _ in range(4):
            for i in range(32):
                result[i] = (
                    result[i] ^ result[(i + 17) % 32] + result[(i + 11) % 32]
                ) & 0xFF

        return result

    def forward(self, password: torch.Tensor, salt: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # For each 32-byte output block i (1-based), computes U1 = PRF(password,
        # salt || [0, 0, 0, i]) then iterates U_j = PRF(password, U_{j-1}) for
        # iterations-1 times, XOR-accumulating into F. Concatenates blocks and returns
        # the first dk_len bytes (int64).
        """
        Derive key from password using PBKDF2.

        Args:
            password: (P,) password bytes
            salt: (S,) salt bytes

        Returns:
            derived_key: (dk_len,) derived key bytes
        """
        device = password.device

        # Number of blocks needed
        num_blocks = (self.dk_len + 31) // 32

        derived_key = torch.zeros(num_blocks * 32, dtype=torch.int64, device=device)

        for block_idx in range(num_blocks):
            # First iteration: U_1 = PRF(Password, Salt || INT(i))
            block_num = torch.tensor(
                [0, 0, 0, block_idx + 1], dtype=torch.int64, device=device
            )
            U = self._simple_hmac(password, torch.cat([salt, block_num]))

            # Accumulator
            F = U.clone()

            # Remaining iterations: U_j = PRF(Password, U_{j-1})
            for _ in range(self.iterations - 1):
                U = self._simple_hmac(password, U)
                F = self._xor(F, U)

            # Store block
            derived_key[block_idx * 32 : (block_idx + 1) * 32] = F

        return derived_key[: self.dk_len]


# Problem configuration
def get_inputs():
    # [FUNCTION DOC]
    # Returns a random 16-byte password and 16-byte salt (int64 byte values).
    password = torch.randint(0, 256, (16,), dtype=torch.int64)  # 16-byte password
    salt = torch.randint(0, 256, (16,), dtype=torch.int64)  # 16-byte salt
    return [password, salt]


def get_init_inputs():
    # [FUNCTION DOC]
    # Returns [1000, 32] (iterations, dk_len).
    return [1000, 32]  # iterations, dk_len
