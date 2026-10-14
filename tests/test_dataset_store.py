import io

import openpyxl
import pandas as pd
import pytest

from app import create_app, db
from app.models import Observation, UploadedFile
from app.services import dataset_store

BASE = [
    "REGIONE", "LABORATORIO", "OSPEDALE", "CODICE_PAZIENTE", "SESSO", "ETA", "NAZIONALITA",
    "PROVINCIA_RESIDENZA", "PAZIENTE_RICOVERATO", "DATA_RICOVERO", "REPARTO_DI_RICOVERO",
    "CODICE_CAMPIONE", "DATA_PRELIEVO", "MATERIALE", "MICROORGANISMO",
]
ABX = ["AMK", "CIP", "COL", "MEM"]
HEADER = BASE + [f"{a}_{kind}" for a in ABX for kind in ("QUALITATIVO", "QUANTITATIVO")]


def _row(patient, sample, date, **tests):
    base = ["190", "IT173", "19092601", patient, "M", "76", "", "82053", "1", "", "49",
            sample, date, "1", "ACISPP"]
    values = []
    for abx in ABX:
        values.extend(tests.get(abx, ("", "")))
    return base + values


ROWS = [
    _row("80025028", "0314324402", "14/03/2022", AMK=("R", ">32"), CIP=("R", ">2"), COL=("S", "<=0.5")),
    _row("80025028", "0221325702", "21/02/2022", AMK=("R", ">32"), COL=("S", "<=0.5")),   # repeat isolate
    _row("00006267", "0112361101", "12/01/2022", AMK=("R", "16"), MEM=("R", ">8.0")),     # leading zeros
    _row("70157604", "0905347402", "05/09/2022", AMK=("S", "8"), COL=("S", "")),
    _row("70157604", "0905347402", "05/09/2022", AMK=("S", "8"), COL=("S", "")),          # exact duplicate
    _row("99999999", "0101010101", "31/02/2022", AMK=("R", ">32")),                        # invalid date
    _row("88888888", "0202020202", "10/10/2022", AMK=("X", "abc"), CIP=("", "4")),        # bad values
]


def _xlsx_bytes(rows=ROWS, header=HEADER) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(header)
    for row in rows:
        ws.append(row)  # strings are stored as text cells, as in the network export
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture
def xlsx_path(tmp_path):
    path = tmp_path / "rete_mic.xlsx"
    path.write_bytes(_xlsx_bytes())
    return path


def test_read_and_validate_keeps_codes_and_parses_mic(xlsx_path):
    raw = dataset_store.read_raw_file(xlsx_path)
    isolates, ast, report, issues = dataset_store.build_and_validate(raw)

    assert report["stats"]["rows_read"] == 7
    assert report["stats"]["isolates_valid"] == 5  # invalid date + exact duplicate excluded
    assert "00006267" in set(isolates["codice_paziente"])
    assert isolates["data_prelievo"].min() == pd.Timestamp("2022-01-12")

    mem = ast[ast["antibiotic"] == "MEM"].iloc[0]
    assert (mem["mic_operator"], mem["mic_value"]) == (">", 8.0)
    col = ast[(ast["antibiotic"] == "COL") & ast["mic_raw"].eq("<=0.5")].iloc[0]
    assert (col["mic_operator"], col["mic_value"]) == ("<=", 0.5)

    problems = set(issues["problema"])
    assert "Data prelievo mancante o non valida: riga esclusa" in problems
    assert "Riga duplicata identica: esclusa" in problems
    assert "Valore qualitativo diverso da S/I/R" in problems
    assert "MIC non interpretabile" in problems
    assert "MIC presente senza interpretazione S/I/R" in problems
    assert report["stats"]["repeat_isolates"] == 1
    assert report["stats"]["first_isolates"] == 4


def test_missing_required_columns_blocks_import(tmp_path):
    header = [c for c in HEADER if c != "MICROORGANISMO"]
    path = tmp_path / "bad.xlsx"
    path.write_bytes(_xlsx_bytes(rows=[r[:14] + r[15:] for r in ROWS], header=header))
    with pytest.raises(dataset_store.DatasetValidationError) as exc:
        dataset_store.build_and_validate(dataset_store.read_raw_file(path))
    assert "MICROORGANISMO" in str(exc.value)


def test_iso_dates_are_not_swapped():
    parsed = dataset_store.parse_dates(pd.Series(["2022-03-04 00:00:00", "04/03/2022", "44634"]))
    assert list(parsed.dt.strftime("%Y-%m-%d")) == ["2022-03-04", "2022-03-04", "2022-03-14"]


def test_upload_stores_dataset_for_models(tmp_path):
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.sqlite'}",
        "UPLOAD_FOLDER": str(tmp_path / "data"),
        "MODEL_FOLDER": str(tmp_path / "models"),
        "DATASET_FOLDER": str(tmp_path / "dataset"),
    })
    client = app.test_client()
    response = client.post(
        "/upload",
        data={"dataset": (io.BytesIO(_xlsx_bytes()), "rete_mic.xlsx")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert "Validazione dati" in response.get_data(as_text=True)

    with app.app_context():
        uploaded = UploadedFile.query.one()
        assert uploaded.status == "imported"
        assert Observation.query.count() > 0
        upload_id = uploaded.id

    folder = tmp_path / "dataset"
    isolates = dataset_store.load_isolates(folder)
    assert len(isolates) == 5
    assert len(dataset_store.load_isolates(folder, first_isolate_only=True)) == 4
    ast = dataset_store.load_ast_results(folder)
    assert {"eta", "sesso", "reparto_di_ricovero", "mic_value", "result"} <= set(ast.columns)

    assert client.get(f"/upload/{upload_id}/validazione").status_code == 200
    assert client.get(f"/upload/{upload_id}/anomalie.csv").status_code == 200

    # Same file again is skipped, not imported twice
    client.post(
        "/upload",
        data={"dataset": (io.BytesIO(_xlsx_bytes()), "rete_mic_copy.xlsx")},
        content_type="multipart/form-data",
    )
    assert len(dataset_store.load_isolates(folder)) == 5
