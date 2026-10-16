#!/usr/bin/env python3
"""Build the fictional Step 1 corpus using only the Python standard library."""

import argparse
import json
import re
import tempfile
import textwrap
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
WARNING = "FICTIONAL CASE - NOT A REAL PATIENT"
CATEGORIES = {"problem", "medication", "allergy", "event", "recommendation"}
TEMPORALITIES = {"recorded", "historical", "unknown"}
SPLITS = {
    "development": [f"dev-{number:02d}" for number in range(1, 13)],
    "held_out": [f"holdout-{number:02d}" for number in range(1, 9)],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, names, context):
    require(isinstance(value, dict) and set(value) == set(names.split()),
            f"Unexpected fields in {context}")


def text(value, context):
    require(isinstance(value, str) and bool(value.strip()), f"Empty text: {context}")
    require(all(character == "\n" or 32 <= ord(character) <= 126 for character in value),
            f"Only printable ASCII and newlines are allowed: {context}")


def identifier(value, context):
    require(isinstance(value, str) and len(value) <= 64
            and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value),
            f"Invalid identifier: {context}")


def optional_date(value):
    if value is not None:
        require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
                f"Invalid ISO date: {value!r}")


def normalize(value):
    return " ".join(value.split())


def page_text(case_id, document, page_number):
    return "\n".join([
        WARNING,
        f"Case: {case_id} | Document: {document['document_id']} | Page: {page_number}",
        document["title"],
        f"Document date: {document['document_date'] or 'not provided'}",
        document["pages"][page_number - 1],
    ])


def display_lines(page):
    lines = []
    for line in page.splitlines():
        lines.extend(textwrap.wrap(line, width=88, break_long_words=False,
                                   break_on_hyphens=False) or [""])
    require(len(lines) <= 52 and all(len(line) <= 88 for line in lines),
            "Page exceeds the printable area")
    return lines


def pdf_literal(value):
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages):
    """PDF 1.4: a standard font, one text stream per page, and a real xref table."""
    require(bool(pages), "A PDF needs at least one page")
    kids = " ".join(f"{4 + 2 * index} 0 R" for index in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
    ]
    for index, page in enumerate(pages):
        text(page, "PDF page")
        lines = display_lines(page)
        commands = ["BT", "/F1 10 Tf", "13 TL", "42 750 Td"]
        for line_number, line in enumerate(lines):
            if line_number:
                commands.append("T*")
            commands.append(f"({pdf_literal(line)}) Tj")
        commands.append("ET")
        stream = ("\n".join(commands) + "\n").encode("ascii")
        objects.append((
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {5 + 2 * index} 0 R >>"
        ).encode("ascii"))
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                       + stream + b"endstream")
    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    result.extend((f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
                   f"startxref\n{xref}\n%%EOF\n").encode("ascii"))
    return bytes(result)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_case(case, split):
    fields(case, "case_id title description split fictional documents facts review_items", "case")
    case_id = case["case_id"]
    identifier(case_id, "case_id")
    require(case["split"] == split and case["fictional"] is True,
            f"Split or fictional marker mismatch: {case_id}")
    text(case["title"], "case title")
    text(case["description"], "case description")
    require(isinstance(case["documents"], list) and 1 <= len(case["documents"]) <= 3,
            f"Expected 1-3 documents: {case_id}")
    sources = {}
    document_ids = set()
    for document in case["documents"]:
        fields(document, "document_id title document_date pages", "document")
        document_id = document["document_id"]
        identifier(document_id, "document_id")
        require(document_id not in document_ids, f"Duplicate document: {case_id}:{document_id}")
        document_ids.add(document_id)
        text(document["title"], "document title")
        optional_date(document["document_date"])
        require(isinstance(document["pages"], list) and bool(document["pages"]), "Missing pages")
        for page_number, body in enumerate(document["pages"], 1):
            text(body, "page body")
            page = page_text(case_id, document, page_number)
            require(len(page) < 1500, f"Page too long: {case_id}:{document_id}:p{page_number}")
            display_lines(page)
            sources[f"{case_id}:{document_id}:p{page_number}:b1"] = normalize(page)
    require(isinstance(case["facts"], list) and bool(case["facts"]), "Missing manual facts")
    fact_ids = set()
    for number, fact in enumerate(case["facts"], 1):
        fields(fact, "id category value event_date temporality evidence", "fact")
        require(fact["id"] == f"fact-{number:02d}", f"Non-sequential fact ID: {case_id}")
        fact_ids.add(fact["id"])
        require(fact["category"] in CATEGORIES, "Invalid category")
        require(fact["temporality"] in TEMPORALITIES, "Invalid temporality")
        text(fact["value"], "fact value")
        optional_date(fact["event_date"])
        require(isinstance(fact["evidence"], list) and bool(fact["evidence"]), "Missing evidence")
        seen = set()
        for evidence in fact["evidence"]:
            fields(evidence, "source_id quote", "evidence")
            source_id = evidence["source_id"]
            text(evidence["quote"], "quote")
            require(source_id in sources, f"Unknown or cross-case source: {source_id}")
            require(evidence["quote"] == normalize(evidence["quote"])
                    and evidence["quote"] in sources[source_id],
                    f"Quote is not contiguous in normalized source: {case_id}:{fact['id']}")
            pair = (source_id, evidence["quote"])
            require(pair not in seen, "Duplicate evidence entry")
            seen.add(pair)
    require(isinstance(case["review_items"], list), "review_items must be a list")
    for item in case["review_items"]:
        fields(item, "reason fact_ids", "review item")
        require(isinstance(item["reason"], str)
                and re.fullmatch(r"[a-z]+(?:_[a-z]+)*", item["reason"]), "Invalid review code")
        require(isinstance(item["fact_ids"], list)
                and len(item["fact_ids"]) == len(set(item["fact_ids"]))
                and set(item["fact_ids"]) <= fact_ids, "Invalid review fact references")


def load_cases():
    cases = []
    for split, case_ids in SPLITS.items():
        path = FIXTURES / "sources" / f"{split}.json"
        batch = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=unique_object)
        require(isinstance(batch, list), f"Expected a list in {path}")
        for case in batch:
            validate_case(case, split)
        require([case["case_id"] for case in batch] == case_ids, f"Unexpected case list: {split}")
        cases.extend(batch)
    return cases


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("ascii")


def build_outputs(cases):
    outputs = {Path("cases/catalog.json"): json_bytes([case["case_id"] for case in cases])}
    for case in cases:
        case_id = case["case_id"]
        manifest = {
            key: case[key] for key in ("case_id", "title", "description", "split", "fictional")
        }
        manifest["documents"] = []
        for document in case["documents"]:
            entry = {key: document[key] for key in ("document_id", "title", "document_date")}
            entry["filename"] = f"{document['document_id']}.pdf"
            manifest["documents"].append(entry)
            pages = [page_text(case_id, document, number)
                     for number in range(1, len(document["pages"]) + 1)]
            outputs[Path("cases") / case_id / entry["filename"]] = make_pdf(pages)
        outputs[Path("cases") / case_id / "manifest.json"] = json_bytes(manifest)
        expected = {key: case[key] for key in ("case_id", "split", "facts", "review_items")}
        outputs[Path("expected") / f"{case_id}.json"] = json_bytes(expected)
    return outputs


def safe_target(root, relative):
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe output path")
    require(not root.is_symlink(), f"Refusing symlink output root: {root}")
    target = root
    for part in relative.parts:
        target = target / part
        require(not target.is_symlink(), f"Refusing symlink: {target}")
    return target


def publish(outputs, root, check=False):
    missing = []
    for relative, content in outputs.items():
        target = safe_target(root, relative)
        if target.exists():
            require(target.is_file() and target.read_bytes() == content,
                    f"Refusing to overwrite differing artifact: {target}")
        else:
            missing.append((relative, content))
    require(
        not check or not missing, f"Missing generated artifacts: {[str(p) for p, _ in missing]}"
    )
    if not check:
        for relative, content in missing:
            target = safe_target(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects against another writer after preflight.
            with target.open("xb") as handle:
                handle.write(content)
    return len(missing)


def self_test():
    """Generator-only checks; no production parser, application, or model is used."""
    class GeneratorChecks(unittest.TestCase):
        def sample(self):
            return json.loads(json.dumps(load_cases()[0]))

        def test_pdf_offsets_stream_lengths_escaping_and_repeatability(self):
            data = make_pdf(["Literal (parentheses) and \\ slash", "Second page"])
            self.assertEqual(data, make_pdf(["Literal (parentheses) and \\ slash", "Second page"]))
            self.assertIn(b"\\(parentheses\\) and \\\\ slash", data)
            xref = int(data.split(b"startxref\n")[1].splitlines()[0])
            self.assertTrue(data[xref:].startswith(b"xref\n"))
            rows = data[xref:].splitlines()
            count = int(rows[1].split()[1])
            for number, row in enumerate(rows[3:count + 2], 1):
                offset = int(row.split()[0])
                self.assertTrue(data[offset:].startswith(f"{number} 0 obj\n".encode("ascii")))
            streams = list(re.finditer(rb"/Length (\d+) >>\nstream\n", data))
            self.assertEqual(len(streams), 2)
            for match in streams:
                end = match.end() + int(match.group(1))
                self.assertTrue(data[end:].startswith(b"endstream\n"))

        def test_corpus(self):
            cases = load_cases()
            self.assertEqual(len(cases), 20)
            self.assertEqual(build_outputs(cases), build_outputs(cases))

        def test_bad_quote_and_cross_case_evidence_rejected(self):
            for field, value in (("quote", "Unsupported invented fact"),
                                 ("source_id", "holdout-01:visit-note:p1:b1")):
                with self.subTest(field=field):
                    case = self.sample()
                    case["facts"][0]["evidence"][0][field] = value
                    with self.assertRaises(ValueError):
                        validate_case(case, "development")

        def test_duplicate_documents_and_bad_review_rejected(self):
            case = self.sample()
            case["documents"].append(case["documents"][0])
            with self.assertRaises(ValueError):
                validate_case(case, "development")
            case = self.sample()
            case["review_items"] = [{"reason": "undated_information", "fact_ids": ["fact-99"]}]
            with self.assertRaises(ValueError):
                validate_case(case, "development")

        def test_non_ascii_long_pages_and_bad_dates_rejected(self):
            for body in ("Non-ASCII: \u00e9", "word " * 300):
                case = self.sample()
                case["documents"][0]["pages"][0] = body
                with self.assertRaises(ValueError):
                    validate_case(case, "development")
            with self.assertRaises(ValueError):
                optional_date("2025-02-30")
            with self.assertRaises(ValueError):
                optional_date("20250101")

        def test_duplicate_json_keys_rejected(self):
            with self.assertRaises(ValueError):
                json.loads('{"case_id": "a", "case_id": "b"}', object_pairs_hook=unique_object)

        def test_publish_is_idempotent_and_never_overwrites(self):
            with tempfile.TemporaryDirectory(prefix="generator-check-", dir=FIXTURES) as directory:
                root = Path(directory)
                outputs = {Path("example.json"): b"{}\n"}
                with self.assertRaises(ValueError):
                    publish(outputs, root, check=True)
                self.assertEqual(publish(outputs, root), 1)
                self.assertEqual(publish(outputs, root), 0)
                self.assertEqual(publish(outputs, root, check=True), 0)
                with self.assertRaises(ValueError):
                    publish({Path("new.json"): b"new", Path("example.json"): b"changed"}, root)
                self.assertFalse((root / "new.json").exists())
                self.assertEqual((root / "example.json").read_bytes(), b"{}\n")
                (root / "alias.json").symlink_to(root / "example.json")
                with self.assertRaises(ValueError):
                    publish({Path("alias.json"): b"{}\n"}, root)
                with self.assertRaises(ValueError):
                    publish({Path("../outside.json"): b"{}\n"}, root)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GeneratorChecks))
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify all outputs without writing")
    mode.add_argument("--self-test", action="store_true", help="Run stdlib generator checks")
    args = parser.parse_args()
    if args.self_test:
        return 0 if self_test() else 1
    try:
        cases = load_cases()
        outputs = build_outputs(cases)
        created = publish(outputs, FIXTURES, check=args.check)
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f"Fixture build failed: {error}\n")
    documents = sum(len(case["documents"]) for case in cases)
    pages = sum(len(document["pages"]) for case in cases for document in case["documents"])
    facts = sum(len(case["facts"]) for case in cases)
    print(
        f"Verified {len(cases)} cases, {documents} documents, {pages} pages, "
        f"{facts} manual facts; {len(outputs)} artifacts, {created} created."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())