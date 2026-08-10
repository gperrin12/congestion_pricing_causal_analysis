.PHONY: data panel estimates report test

PYTHON ?= python
export PYTHONPATH := src

data:
	$(PYTHON) -m cpca.ingest.download_mta_ridership
	$(PYTHON) -m cpca.ingest.download_noaa_weather
	$(PYTHON) -m cpca.ingest.download_crz_vehicle_entries
	$(PYTHON) -m cpca.ingest.download_bt_crossings
	$(PYTHON) -m cpca.geo.build_crz_polygon
	$(PYTHON) -m cpca.geo.assign_zones

panel:
	$(PYTHON) -m cpca.panel.build_station_week --force
	$(PYTHON) -m cpca.panel.build_bt_day --force
	$(PYTHON) -m cpca.panel.build_crz_vehicle_day --force
	$(PYTHON) -m pytest tests/test_panel_invariants.py -q

estimates:
	$(PYTHON) -m cpca.estimators.bsts
	$(PYTHON) -m cpca.estimators.did --treated-zone crz
	$(PYTHON) -m cpca.estimators.did --treated-zone border
	$(PYTHON) -m cpca.inference.placebo_did --treated-zone crz

report:
	@command -v quarto >/dev/null 2>&1 || { \
		echo "quarto not found; install from https://quarto.org/docs/get-started/"; \
		exit 1; \
	}
	cd report && quarto render

test:
	$(PYTHON) -m pytest tests -q
