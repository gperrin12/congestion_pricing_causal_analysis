.PHONY: data panel estimates report test

PYTHON ?= python
export PYTHONPATH := src

data:
	$(PYTHON) -m cpca.ingest.download_mta_ridership
	$(PYTHON) -m cpca.ingest.download_noaa_weather
	$(PYTHON) -m cpca.ingest.download_crz_vehicle_entries
	$(PYTHON) -m cpca.geo.build_crz_polygon
	$(PYTHON) -m cpca.geo.assign_zones

panel:
	$(PYTHON) -m cpca.panel.build_station_week --force
	$(PYTHON) -m pytest tests/test_panel_invariants.py -q

estimates:
	@echo "Estimator grid not wired yet"

report:
	@command -v quarto >/dev/null 2>&1 || { \
		echo "quarto not found; install from https://quarto.org/docs/get-started/"; \
		exit 1; \
	}
	cd report && quarto render

test:
	$(PYTHON) -m pytest tests -q
