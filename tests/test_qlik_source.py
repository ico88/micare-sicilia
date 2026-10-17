import csv

import pytest

from app import create_app, db
from app.models import UploadedFile
from app.services import dashboard_jobs, dataset_store, jobs, qlik_source
from app.services.importing import import_file

FIELDS = ["KEY_CAMPIONE", "REGIONE", "LABORATORIO", "OSPEDALE", "CODICE_PAZIENTE", "SESSO", "ETA",
          "PAZIENTE_RICOVERATO", "REPARTO_DI_RICOVERO", "CODICE_CAMPIONE", "DATA_PRELIEVO", "MATERIALE",
          "MICROORGANISMO", "AMK_QUALITATIVO", "AMK_QUANTITATIVO", "MEM_QUALITATIVO", "MEM_QUANTITATIVO",
          "ETA_PZ", "ANNO_PRELIEVO"]


def _record(i, material="1 - Sangue", date="14/03/2023 00:00:00"):
    return {
        "KEY_CAMPIONE": f"K{i:04d}", "REGIONE": "190", "LABORATORIO": "IT173",
        "OSPEDALE": "$ - NA" if i % 2 else "190201 - OSP. CIVICO", "CODICE_PAZIENTE": f"{i:08d}",
        "SESSO": "M", "ETA": "76", "PAZIENTE_RICOVERATO": "1 - \tricovero ordinario",
        "REPARTO_DI_RICOVERO": "49 - Medicina", "CODICE_CAMPIONE": f"03{i:08d}", "DATA_PRELIEVO": date,
        "MATERIALE": material, "MICROORGANISMO": "KLEPNE", "AMK_QUALITATIVO": "R", "AMK_QUANTITATIVO": ">32",
        "MEM_QUALITATIVO": None, "MEM_QUANTITATIVO": None, "ETA_PZ": "76", "ANNO_PRELIEVO": "2023",
    }


class FakeEngine:
    """In-memory stand-in for the Qlik Engine API, enough for the extractor."""

    def __init__(self, records, declared_rows=None, null_key_row=True):
        self.app_id = "app"
        self.records = records
        self.declared_rows = len(records) if declared_rows is None else declared_rows
        self.null_key_row = null_key_row
        self.objects, self.answers, self.destroyed = {}, {}, []
        self._id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def send(self, method, handle, params):
        self._id += 1
        self.answers[self._id] = self._answer(method, handle, params)
        return self._id

    def receive(self, ids):
        return {i: self.answers.pop(i) for i in reversed(ids)}  # arrival order differs from page order

    def call(self, method, handle, params):
        return self.receive([self.send(method, handle, params)])[self._id]

    def _answer(self, method, handle, params):
        if method == "OpenDoc":
            return {"qReturn": {"qHandle": 1}}
        if method == "GetTablesAndKeys":
            return {"qtr": [{"qName": "FATTORI", "qNoOfRows": 3, "qFields": [{"qName": "KEY_ANTIBIOTICO"}]},
                            {"qName": "Dati_Grezzi", "qNoOfRows": self.declared_rows,
                             "qFields": [{"qName": f} for f in FIELDS]}]}
        if method == "CreateSessionObject":
            fields = [d["qDef"]["qFieldDefs"][0] for d in params[0]["qHyperCubeDef"]["qDimensions"]]
            rows = sorted(self.records, key=lambda r: r["KEY_CAMPIONE"])
            matrix = [[self._cell(r[f]) for f in fields] for r in rows]
            if self.null_key_row:  # key-less combination coming from an associated table
                matrix.insert(0, [{"qText": "-", "qIsNull": True}] + [{"qText": "-", "qIsNull": True}] * (len(fields) - 1))
            handle = 10 + len(self.objects)
            self.objects[handle] = matrix
            return {"qReturn": {"qHandle": handle}}
        if method == "GetLayout":
            return {"qLayout": {"qHyperCube": {"qSize": {"qcy": len(self.objects[handle])}}}}
        if method == "GetHyperCubeData":
            page = params[1][0]
            return {"qDataPages": [{"qMatrix": self.objects[handle][page["qTop"]:page["qTop"] + page["qHeight"]]}]}
        if method == "DestroySessionObject":
            self.destroyed.append(params[0])
        return {}

    @staticmethod
    def _cell(value):
        return {"qText": "-", "qIsNull": True} if value is None else {"qText": value}


@pytest.fixture
def small_pages(monkeypatch):
    monkeypatch.setattr(qlik_source, "MAX_CELLS", 40)       # several pages per block
    monkeypatch.setattr(qlik_source, "FIELDS_PER_CUBE", 5)  # several blocks
    monkeypatch.setattr(qlik_source, "IN_FLIGHT", 2)


def _extract(tmp_path, records, **engine_options):
    engine = FakeEngine(records, **engine_options)
    out = tmp_path / "cruscotto.csv"
    stats = qlik_source.extract_to_csv("https://x/anonimo/single/?appid=app", out,
                                       engine_factory=lambda url: engine)
    return out, stats, engine


def test_extraction_writes_the_export_layout(tmp_path, small_pages):
    records = [_record(i) for i in range(1, 8)] + [_record(8, material="3 - Urine")]
    out, stats, engine = _extract(tmp_path, records)

    assert stats["rows"] == 8
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    assert list(rows[0]) == [f for f in FIELDS if f not in qlik_source.DERIVED_FIELDS]
    assert [r["CODICE_PAZIENTE"] for r in rows] == [f"{i:08d}" for i in range(1, 9)]  # leading zeros kept
    first, second = rows[0], rows[1]
    assert first["MATERIALE"] == "1" and rows[-1]["MATERIALE"] == "3"
    assert first["OSPEDALE"] == "" and second["OSPEDALE"] == "190201"
    assert first["PAZIENTE_RICOVERATO"] == "1" and first["REPARTO_DI_RICOVERO"] == "49"
    assert first["MEM_QUALITATIVO"] == "" and first["AMK_QUANTITATIVO"] == ">32"
    assert len(engine.destroyed) == 4  # one session object per block, all released

    isolates, ast, report, _ = dataset_store.build_and_validate(dataset_store.read_raw_file(out))
    assert report["stats"]["isolates_valid"] == 8
    assert isolates["data_prelievo"].dt.year.eq(2023).all()


def test_incomplete_extraction_is_refused(tmp_path, small_pages):
    with pytest.raises(qlik_source.QlikSourceError, match="incompleta"):
        _extract(tmp_path, [_record(1), _record(2)], declared_rows=3)
    assert not (tmp_path / "cruscotto.csv").exists()


def test_dashboard_url_parts():
    assert qlik_source.parse_dashboard_url(qlik_source.DASHBOARD_URL) == (
        "https://qlik.qualitasiciliassr.it", "/anonimo", "85ada16c-4b41-4bc6-9ca1-405b8243d0c2")
    with pytest.raises(qlik_source.QlikSourceError):
        qlik_source.parse_dashboard_url("https://example.org/qualcosa")


@pytest.fixture
def app(tmp_path):
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.sqlite'}",
        "UPLOAD_FOLDER": str(tmp_path / "data"),
        "MODEL_FOLDER": str(tmp_path / "models"),
        "DATASET_FOLDER": str(tmp_path / "dataset"),
        "FORECAST_FOLDER": str(tmp_path / "previsioni"),
    })
    yield app
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def test_dashboard_refresh_replaces_instead_of_adding(app, tmp_path, small_pages, monkeypatch):
    manual, _, _ = _extract(tmp_path, [_record(i) for i in range(1, 5)])
    with app.app_context():
        first = import_file(manual, "export_manuale.csv", app.config["DATASET_FOLDER"])

    records = [_record(i) for i in range(1, 7)]
    monkeypatch.setattr(dashboard_jobs, "extract_to_csv", lambda url, path, progress: qlik_source.extract_to_csv(
        url, path, progress, engine_factory=lambda u: FakeEngine(records)))
    for _ in range(2):  # a second refresh must replace the first one as well
        assert dashboard_jobs.start_dashboard_job(app, include_manual=True) is None
        jobs.wait("cruscotto", timeout=30)
        job = jobs.get("cruscotto")
        assert job["status"] == "success", job["error"]
        upload_id = job["link_args"]["upload_id"]

    with app.app_context():
        imported = UploadedFile.query.filter_by(status="imported").all()
        assert [u.id for u in imported] == [upload_id]
        assert imported[0].original_name.startswith(dashboard_jobs.FILE_PREFIX)
        assert db.session.get(UploadedFile, first.upload_id).status == "replaced"
        assert len(dataset_store.load_isolates(app.config["DATASET_FOLDER"])) == 6
        # only the latest download stays on disk; the file uploaded by hand is not touched
        downloads = sorted(p.name for p in (tmp_path / "data").glob(f"{dashboard_jobs.FILE_PREFIX}*"))
        assert downloads == [imported[0].original_name] and manual.exists()

        page = app.test_client().get("/").get_data(as_text=True)
        assert "Scarica dal cruscotto" in page and "completato" in page


def test_manual_uploads_are_kept_when_asked(app, tmp_path, small_pages):
    manual, _, _ = _extract(tmp_path, [_record(i) for i in range(1, 3)])
    with app.app_context():
        import_file(manual, "export_manuale.csv", app.config["DATASET_FOLDER"])
        assert dashboard_jobs.uploads_to_replace(include_manual=False) == []
        page = app.test_client().get("/").get_data(as_text=True)
        assert "export_manuale.csv" in page and 'name="sostituisci_manuali"' in page


def test_failed_refresh_leaves_the_archive_alone(app, tmp_path, small_pages, monkeypatch):
    manual, _, _ = _extract(tmp_path, [_record(i) for i in range(1, 3)])
    with app.app_context():
        first = import_file(manual, "export_manuale.csv", app.config["DATASET_FOLDER"])

    def broken(url, path, progress):
        raise qlik_source.QlikSourceError("Cruscotto non raggiungibile: prova")

    monkeypatch.setattr(dashboard_jobs, "extract_to_csv", broken)
    assert dashboard_jobs.start_dashboard_job(app, include_manual=True) is None
    jobs.wait("cruscotto", timeout=30)
    assert jobs.get("cruscotto")["status"] == "error"
    with app.app_context():
        assert [u.id for u in UploadedFile.query.filter_by(status="imported")] == [first.upload_id]
        assert "non riuscito" in app.test_client().get("/").get_data(as_text=True)
