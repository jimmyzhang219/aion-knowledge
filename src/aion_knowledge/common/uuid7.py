"""UUIDv7 生成器 —— 时间有序 UUID，兼容 Python < 3.14。

UUIDv7 格式（RFC 9562）：
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                           unix_ts_ms                          |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |          unix_ts_ms           |  ver  |       random_a        |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |var|                      random_b                             |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                           random_b                            |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
"""

import random
import time
import uuid


def uuid7() -> uuid.UUID:
    """生成一个 UUIDv7（时间有序）。"""
    # unix_ts_ms: 48-bit timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000)

    # Random data: 74 bits
    rand_a = random.getrandbits(12)   # 12 bits
    rand_b = random.getrandbits(62)   # 62 bits

    # Build the 128-bit integer
    # Bytes layout:
    #   [0-5]: timestamp_ms (48 bits)
    #   [6]:   (rand_a >> 8) & 0x0F | 0x70  (version 7)
    #   [7]:   rand_a & 0xFF
    #   [8]:   (rand_b >> 56) & 0x3F | 0x80  (variant 1)
    #   [9-15]: remaining 56 bits of rand_b

    uuid_int = (
        (timestamp_ms & 0xFFFFFFFFFFFF) << 80
        | (rand_a & 0xFFF) << 64
        | (rand_b & 0x3FFFFFFFFFFFFFFF) << 0
    )
    # Set version (7) and variant (RFC 4122)
    uuid_int &= ~(0xF000 << 64)          # Clear version nibble
    uuid_int |= 0x7000 << 64             # Set version to 7
    uuid_int &= ~(0xC000 << 48)          # Clear variant bits
    uuid_int |= 0x8000 << 48             # Set variant to RFC 4122

    return uuid.UUID(int=uuid_int)
