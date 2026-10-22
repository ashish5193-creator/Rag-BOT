import unittest

from app.services.documents import DocumentError, _extract_text


class DocumentParsingTests(unittest.TestCase):
    def test_text_extension_is_case_insensitive(self) -> None:
        self.assertEqual(_extract_text("notes.TXT", b"\xef\xbb\xbfHello"), "Hello")

    def test_unsupported_extension_is_rejected(self) -> None:
        with self.assertRaisesRegex(DocumentError, "PDF or TXT"):
            _extract_text("notes.docx", b"content")

    def test_empty_upload_is_rejected(self) -> None:
        with self.assertRaisesRegex(DocumentError, "empty"):
            _extract_text("notes.txt", b"")

    def test_invalid_text_encoding_is_rejected(self) -> None:
        with self.assertRaisesRegex(DocumentError, "Could not read"):
            _extract_text("notes.txt", b"\xff")

    def test_invalid_pdf_is_rejected(self) -> None:
        with self.assertRaisesRegex(DocumentError, "Could not read"):
            _extract_text("notes.pdf", b"not a PDF")


if __name__ == "__main__":
    unittest.main()
