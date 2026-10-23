PYTHON ?= python3

.PHONY: help data tune eval eval-inner eval-main final final-personal final-hybrid analysis cases verify report all clean-cache

help:
	@echo "SWEET.TV midterm pipeline"
	@echo "  make data       rebuild sessions, EPG cache, clean overlaps"
	@echo "  make tune       tune hybrid weights on the inner split"
	@echo "  make eval       inner + main validation tables"
	@echo "  make final      fit the selected personal model, write submission.csv"
	@echo "  make analysis   bootstrap CIs, overlap, cold-start, figures"
	@echo "  make cases      fixed-user case-study tables"
	@echo "  make verify     hand checks and submission format validation"
	@echo "  make report     PDF + Markdown report"
	@echo "  make all        full rebuild in dependency order"
	@echo "  make clean-cache remove generated parquet caches"

data:
	$(PYTHON) midterm/data_prep.py --force

tune:
	$(PYTHON) midterm/evaluate.py --split inner --tune

eval-inner:
	$(PYTHON) midterm/evaluate.py --split inner

eval-main:
	$(PYTHON) midterm/evaluate.py --split main

eval: eval-inner eval-main

final final-personal:
	$(PYTHON) midterm/run_final_personal.py --variant repeat --output submission.csv

final-hybrid:
	$(PYTHON) midterm/run_final.py

analysis:
	$(PYTHON) midterm/analysis.py

cases:
	$(PYTHON) midterm/case_studies.py

verify:
	$(PYTHON) midterm/verify.py

report:
	$(PYTHON) midterm/report.py

all: data tune eval-main final analysis cases verify report
	@echo "Done: submission.csv and midterm_report/midterm_report.pdf"

clean-cache:
	rm -f midterm_artifacts/sessions.parquet midterm_artifacts/epg_all.parquet \
	      midterm_artifacts/overlaps_clean.parquet midterm_artifacts/overlaps_airings.parquet \
	      midterm_artifacts/show_metadata.parquet
