import json
import tempfile
import unittest
from pathlib import Path

from spectrashift.retrieval import load_evidence_documents, query_evidence


class EvidenceRetrievalTests(unittest.TestCase):
    def test_retrieves_cited_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text(
                "# Pilot\n\n## Failure analysis\n\n"
                "PCA-logistic has poor calibration and zero goethite true positives.\n",
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "decision": {"deployment_ready": False},
                        "models": {"prototype-sam": {"macro_f1": 0.199}},
                    }
                ),
                encoding="utf-8",
            )
            documents = load_evidence_documents(root)
            self.assertGreaterEqual(len(documents), 3)
            result = query_evidence(
                root,
                "Why is PCA logistic not deployment ready?",
                top_k=2,
            )
            self.assertEqual(result["retrieval_backend"], "tfidf-cosine")
            self.assertTrue(result["hits"])
            self.assertIn("README.md#failure-analysis", result["hits"][0]["citation"])
            self.assertIn("cannot change predictions", result["answer_policy"])


if __name__ == "__main__":
    unittest.main()
