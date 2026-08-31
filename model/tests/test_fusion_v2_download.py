from __future__ import annotations

import unittest
from collections import Counter

from scripts.download_fusion_v2_data import claim_unique_digest


class FusionV2DownloadTests(unittest.TestCase):
    def test_duplicate_digest_is_skipped_and_audited(self):
        seen: set[str] = set()
        audit: Counter[str] = Counter()
        self.assertTrue(claim_unique_digest("a" * 64, seen, audit, "openfake"))
        self.assertFalse(claim_unique_digest("a" * 64, seen, audit, "openfake"))
        self.assertTrue(claim_unique_digest("b" * 64, seen, audit, "tigas"))
        self.assertEqual(seen, {"a" * 64, "b" * 64})
        self.assertEqual(audit, {"openfake": 1})


if __name__ == "__main__":
    unittest.main()
