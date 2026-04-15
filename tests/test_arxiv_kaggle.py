from __future__ import annotations

from papersearch.arxiv_kaggle import record_to_paper_dict


def test_record_to_paper_dict_filters_categories():
    rec = {"categories": "math.CO", "title": "x", "abstract": "y" * 20}
    assert record_to_paper_dict(rec) is None


def test_record_to_paper_dict_accepts_cs_ai():
    r = record_to_paper_dict(
        {
            "id": "2301.00001",
            "title": "  Neural nets  ",
            "abstract": "We study deep learning. " * 5,
            "authors": "Alice; Bob",
            "categories": "cs.AI cs.LG",
            "update_date": "2023-05-01",
            "doi": "10.1000/test",
            "journal-ref": "My Conf 2023",
        }
    )
    assert r is not None
    assert r["paper_id"] == "2301.00001"
    assert "cs.AI" in r["category"]
    assert r["year"] == 2023
