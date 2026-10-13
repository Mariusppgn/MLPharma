import http.client
import json
import threading

import numpy as np
import pandas as pd
import pytest

from deep_pharma.combination_profiles.api import make_server
from deep_pharma.combination_profiles.datastore import ProfileStore, sha256
from deep_pharma.combination_profiles.offline import generate_offline, prepare_snapshot
from deep_pharma.combination_profiles.statistics import (
    cell,
    compare_effect,
    filter_rows,
    presence,
    summarize,
)
from deep_pharma.combination_profiles.structures import molecular_structure
from deep_pharma.graph.preprocessing import molecule_id
from deep_pharma.text_features import TEXT_FIELDS


def documented(value, kind="probability", **extra):
    return cell(
        value,
        kind=kind,
        source="documented research source",
        comparability_id="study1",
        meaning="defined event",
        population="cohort1",
        period="period1",
        **extra,
    )


@pytest.mark.parametrize("a,b,ref", [(0, 0, 0), (1, 0, 1), (1, 1, 1), (0.2, 0.3, 0.44)])
def test_probability_union_and_symmetry(a, b, ref):
    r = compare_effect("X", documented(a), documented(b), documented(0.5), independence=True)
    swapped = compare_effect("X", documented(b), documented(a), documented(0.5), independence=True)
    assert r["reference"] == pytest.approx(ref)
    assert r["delta"] == swapped["delta"]
    assert r["reference"] == swapped["reference"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -2, ""])
def test_missing_prr_is_never_zero(value):
    assert cell(value, kind="prr")["value"] is None
    assert presence(cell(value, kind="prr"), 0) is None


def test_prr_has_no_union_even_if_documents_match():
    r = compare_effect(
        "X",
        documented(2, "prr"),
        documented(4, "prr"),
        documented(8, "prr"),
        threshold=0,
        independence=True,
    )
    assert r["reference"] is r["delta"] is None
    assert r["prr_deltas"] == [6, 4]
    assert r["prr_log_ratios"] == pytest.approx([np.log(4), np.log(2)])
    assert set(r["categories"]) == {"shared", "strengthened"}
    zero = compare_effect(
        "X", documented(0, "prr"), documented(4, "prr"), documented(0, "prr"), threshold=0
    )
    assert zero["a"]["value"] == 0 and zero["prr_log_ratios"] is None


def test_independence_requires_explicit_agreement_and_context():
    cells = [documented(0.2), documented(0.3), documented(0.4)]
    assert compare_effect("X", *cells)["reference"] is None
    cells[1]["population"] = "other cohort"
    assert compare_effect("X", *cells, independence=True)["reference"] is None
    cells[1].pop("population")
    assert compare_effect("X", *cells, independence=True)["reference"] is None


def test_annotations_are_not_probabilities():
    cells = [documented(1, "annotation")] * 3
    assert compare_effect("X", *cells, independence=True)["reference"] is None
    with pytest.raises(ValueError):
        cell(1.4, kind="probability")
    with pytest.raises(ValueError):
        cell(0.4, kind="annotation")


def test_emerging_needs_known_individuals_and_can_overlap_strengthened():
    r = compare_effect("X", documented(0.1), documented(0.1), documented(0.9), independence=True)
    assert set(r["categories"]) == {"emerging", "strengthened"}
    missing = compare_effect("X", cell(), documented(0.1), documented(0.9), independence=True)
    assert "emerging" not in missing["categories"]
    low_n = compare_effect(
        "X",
        documented(0.1, n=1),
        documented(0.1, n=100),
        documented(0.9, n=100),
        min_support=10,
        independence=True,
    )
    assert low_n["categories"] == []


def test_combined_filters_and_unknowns():
    positive = compare_effect(
        "Headache",
        documented(0.1),
        documented(0.1),
        documented(0.9, n=40, ci=[0.8, 0.95]),
        independence=True,
    )
    positive.update(label="Céphalée", soc="documented SOC", pt="documented PT")
    negative = compare_effect(
        "Other", documented(0.9), documented(0.9), documented(0.1), independence=True
    )
    unknown = compare_effect("Missing", cell(), cell(), cell())
    rows = [negative, positive, unknown]
    filters = {
        "search": "céph",
        "minimum": 0.8,
        "maximum": 1,
        "delta_sign": "positive",
        "category": "emerging",
        "minimum_n": 30,
        "soc": "documented SOC",
        "pt": "documented PT",
        "max_ci_width": 0.2,
    }
    assert filter_rows(rows, filters) == [positive]
    assert filter_rows(rows, {"minimum": 0.2}) == [positive]
    assert filter_rows(rows, {"delta_sign": "negative"}) == [negative]
    assert summarize([unknown])["detected"] == [None, None, None]
    assert summarize([unknown])["jaccard"] is None


@pytest.fixture
def store(tmp_path):
    pairs = [
        ("CC", "O", 1, 0),
        ("O", "CC", 0, 0),
        ("CC", "N", 1, 0),
        ("CC", "N", 1, 0),
        ("CC", "Cl", 0, 1),
        ("O", "N", 0, 1),
        ("O", "Cl", 1, 0),
    ]
    rows = []
    for i, (a, b, x, y) in enumerate(pairs):
        row = {
            "Pair_ID": f"P{i}",
            "SMILES_A": a,
            "SMILES_B": b,
            "Target_Binary_Headache": x,
            "Target_Binary_Nausea": y,
            "Target_PRR_Headache": x * 3,
            "Target_PRR_Nausea": y * 2,
        }
        for side in ("A", "B"):
            for field in TEXT_FIELDS:
                row[f"{field}_{side}"] = "CYP3A4 documented text" if field == "Metabolism" else ""
        rows.append(row)
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return ProfileStore(path)


def req(**extra):
    return {"a": molecule_id("CC"), "b": molecule_id("O"), **extra}


def test_context_pair_exclusion_deduplication_and_symmetry(store):
    result = store.compare(req(context=True))
    r = result["rows"][0]
    assert r["a"]["value"] == 0.5 and r["a"]["n"] == 2
    assert r["b"]["value"] == 0.5 and r["b"]["n"] == 2
    assert r["reference"] is None and r["delta"] is None
    assert "emerging" not in r["categories"]
    other = store.compare({**req(context=True), "a": molecule_id("O"), "b": molecule_id("CC")})
    for left, right in zip(result["rows"], other["rows"], strict=True):
        assert left["a"] == right["b"] and left["ab"] == right["ab"]
    assert result["selection"]["available_pairs"] == ["P0", "P1"]
    assert result["pharmacology"]["shared_cyp_mentions"] == ["3A4"]


def test_conflicting_context_duplicates_are_missing_not_averaged(store):
    store.binary[3, 0] = 0
    r = store.compare(req(context=True))["rows"][0]
    assert r["a"]["value"] == 0 and r["a"]["n"] == 1


def test_pair_variants_explicit_absent_combo_and_no_monotherapy_inference(store):
    assert store.compare(req(pair_id="P1"))["rows"][0]["ab"]["value"] == 0
    assert store.compare(req())["rows"][0]["a"]["value"] is None
    absent = store.compare({"a": molecule_id("N"), "b": molecule_id("Cl"), "context": True})
    assert absent["selection"]["pair_id"] is None
    assert all(r["ab"]["value"] is None for r in absent["rows"])
    self_pair = store.compare({"a": molecule_id("CC"), "b": molecule_id("CC")})
    assert all(r["a"]["value"] is None for r in self_pair["rows"])
    with pytest.raises(ValueError):
        store.compare(req(pair_id="other"))


def test_imported_documented_profiles_and_missing_effects(store, tmp_path):
    a, b = molecule_id("CC"), molecule_id("O")
    records = [
        {"molecule_a": left, "molecule_b": right, "effect": "Headache", **documented(v)}
        for left, right, v in [(a, "", 0.1), (b, "", 0.1), (a, b, 0.9)]
    ]
    path = tmp_path / "profiles.csv"
    pd.DataFrame(records).drop(columns="ci", errors="ignore").to_csv(path, index=False)
    imported = ProfileStore(store.train, profiles=path)
    result = imported.compare(req(source="imported_probability", independence=True))
    assert result["rows"][0]["reference"] == pytest.approx(0.19)
    assert "emerging" in result["rows"][0]["categories"]
    assert result["rows"][1]["reference"] is None
    assert imported.import_hash == sha256(path)


def test_oof_fail_closed_on_stale_signature(store, tmp_path):
    (tmp_path / "feature_signature.json").write_text('{"train_sha256":"stale"}')
    imported = ProfileStore(store.train, oof_dir=tmp_path, manifest=tmp_path / "fold.csv")
    assert imported.oof is None and "SHA" in imported.oof_status


def test_partner_catalog_is_exact_and_symmetric(store):
    partners = store.catalog()["partners"]
    assert set(partners[molecule_id("N")]) == {molecule_id("CC"), molecule_id("O")}
    for a, values in partners.items():
        for b in values:
            assert a in partners[b]
            assert tuple(sorted((a, b))) in store.pair_rows
    assert molecule_id("Cl") not in partners[molecule_id("N")]


@pytest.mark.parametrize(
    "smiles,atoms,bonds,fragments,charge",
    [
        ("CCO", 3, 2, 1, 0),
        ("c1ccccc1", 6, 6, 1, 0),
        ("[Na+].[Cl-]", 2, 0, 2, 0),
        ("[H+]", 1, 0, 1, 1),
        ("N[C@@H](C)C(=O)O", 6, 5, 1, 0),
    ],
)
def test_structural_svg_preserves_topology(smiles, atoms, bonds, fragments, charge):
    from xml.etree import ElementTree

    from deep_pharma.validation import canonicalize_smiles

    structure = molecular_structure(smiles)
    root = ElementTree.fromstring(structure["svg"])
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib["viewBox"] == "0 0 640 400"
    assert structure["smiles"] == canonicalize_smiles(smiles)
    assert (
        structure["atoms"],
        structure["bonds"],
        structure["fragments"],
        structure["formal_charge"],
    ) == (atoms, bonds, fragments, charge)
    assert root.findall(".//{http://www.w3.org/2000/svg}path")
    assert "<script" not in structure["svg"]


def test_drawings_follow_a_b_and_offline_export(store):
    r = store.compare(req())
    inverse = store.compare({"a": req()["b"], "b": req()["a"]})
    assert r["pharmacology"]["a"]["structure"] == inverse["pharmacology"]["b"]["structure"]
    data = prepare_snapshot(store)
    for side in ("a", "b"):
        molecule = r["pharmacology"][side]
        assert data["molecules"][molecule["id"]]["structure"] == molecule["structure"]


def test_offline_snapshot_context_subtraction_equals_backend(store):
    # Include one inconsistent duplicate: it must contribute neither n nor k for that effect.
    store.binary[3, 0] = 0
    data = prepare_snapshot(store)
    for key, indices in store.pair_rows.items():
        vote = data["pairs"]["\x1f".join(key)]["vote"]
        for mid in key:
            for e, (freq, count, positives) in enumerate(store.context(mid, key)):
                n = data["totals"][mid]["n"][e] - int(vote[e] is not None)
                k = data["totals"][mid]["k"][e] - (vote[e] or 0)
                assert (n, k) == (count, positives)
                assert (k / n if n else None) == freq
        assert data["pairs"]["\x1f".join(key)]["rows"] == indices
    assert data["rows"][0]["prr"] == [3, 0]


def test_generated_html_self_contained_and_payload_roundtrip(store, tmp_path):
    import base64
    import gzip
    import re

    output = tmp_path / "standalone.html"
    report = generate_offline(store, output)
    html = output.read_text(encoding="utf-8")
    assert "<script src=" not in html and '<link rel="stylesheet"' not in html
    assert "connect-src 'none'" in html
    assert 'list="molecules-b"' in html and 'list="molecules-a"' in html
    encoded = re.search(r'id="deep-pharma-snapshot"[^>]*>(.*?)</script>', html).group(1)
    data = json.loads(gzip.decompress(base64.b64decode(encoded)))
    assert data["catalog"]["audit"]["train_sha256"] == store.train_hash
    assert len(data["rows"]) == len(store.frame)
    assert report["canonical_pairs"] == len(store.pair_rows)


def test_api_assets_comparison_and_security(store):
    server = make_server(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection("127.0.0.1", server.server_port)
    try:
        for path in ("/", "/app.js", "/style.css", "/api/catalog"):
            client.request("GET", path)
            response = client.getresponse()
            assert response.status == 200 and response.read()
            assert "frame-ancestors 'none'" in response.getheader("Content-Security-Policy")
        request = req(context=True, filters={"search": "Headache"})
        client.request(
            "POST", "/api/compare", json.dumps(request), {"Content-Type": "application/json"}
        )
        response = client.getresponse()
        result = json.loads(response.read())
        assert response.status == 200 and len(result["rows"]) == result["summary"]["rows"] == 1
        assert result["rows"][0]["effect"] == "Headache"
        client.request("GET", "/api/catalog", headers={"Host": "attacker.example"})
        assert client.getresponse().status == 403
        client.close()
        client = http.client.HTTPConnection("127.0.0.1", server.server_port)
        client.request(
            "POST", "/api/compare", json.dumps(request), {"Origin": "https://attacker.example"}
        )
        response = client.getresponse()
        assert response.status == 403
        response.read()
        client.request("GET", "/../../data/raw/train.csv")
        assert client.getresponse().status == 404
    finally:
        client.close()
        server.shutdown()
        server.server_close()
