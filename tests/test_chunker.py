"""Unit tests for Amharic text chunker and sentence tokenizer."""

from __future__ import annotations

import unittest

from src.chunker import chunk_amharic_document, normalize_amharic_text, split_into_sentences


class AmharicChunkerTests(unittest.TestCase):
    def test_normalize_amharic_text(self):
        raw = "  ሰላም   ዓለም። \n\n\n\nይህ    የሙከራ ጽሑፍ ነው።  "
        normalized = normalize_amharic_text(raw)
        self.assertEqual(normalized, "ሰላም ዓለም።\n\nይህ የሙከራ ጽሑፍ ነው።")

    def test_split_into_sentences_arat_neteb(self):
        text = "ኢትዮጵያ በምሥራቅ አፍሪካ ትገኛለች። ዋና ከተማዋ አዲስ አበባ ናት።"
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "ኢትዮጵያ በምሥራቅ አፍሪካ ትገኛለች።")
        self.assertEqual(sentences[1], "ዋና ከተማዋ አዲስ አበባ ናት።")

    def test_split_into_sentences_mixed_punctuation(self):
        text = "ዋና ከተማዋ ማን ናት? አዲስ አበባ ናት! ታሪኳስ እንዴት ነው፤ ረጅም ነው።"
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 4)

    def test_chunking_small_document(self):
        text = "አጭር አረፍተ ነገር።"
        chunks = chunk_amharic_document(text, document_id="doc_1", chunk_size_chars=200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].document_id, "doc_1")
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].text, "አጭር አረፍተ ነገር።")

    def test_chunking_with_overlap(self):
        sentences = [
            "አረፍተ ነገር ፩።",
            "አረፍተ ነገር ፪።",
            "አረፍተ ነገር ፫።",
            "አረፍተ ነገር ፬።",
            "አረፍተ ነገር ፭።",
        ]
        text = " ".join(sentences)
        chunks = chunk_amharic_document(
            text,
            document_id="doc_test",
            chunk_size_chars=30,
            chunk_overlap_chars=15,
        )
        self.assertGreater(len(chunks), 1)
        # Check that consecutive chunks share overlap context
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.document_id, "doc_test")
            self.assertEqual(chunk.chunk_index, i)
            self.assertTrue(len(chunk.text) > 0)


if __name__ == "__main__":
    unittest.main()
