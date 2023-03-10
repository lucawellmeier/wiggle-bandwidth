BUILD_DIR=build
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/latextmp
	mkdir -p $(BUILD_DIR)/pdf
	mkdir -p $(BUILD_DIR)/results
	mkdir -p $(BUILD_DIR)/figures

COMPUTE_SCRIPTS = $(shell find experiments -name "*.compute.py")
RESULTS = $(patsubst experiments/%.compute.py,$(BUILD_DIR)/results/%.npz,$(COMPUTE_SCRIPTS))
.PRECIOUS: $(RESULTS)
$(BUILD_DIR)/results/%.npz: experiments/%.compute.py | $(BUILD_DIR)
	.venv/bin/python $<

PRESENT_SCRIPTS = $(shell find experiments -name "*.present.py")
FIGURES = $(patsubst experiments/%.present.py,$(BUILD_DIR)/figures/%.png,$(PRESENT_SCRIPTS))
$(BUILD_DIR)/figures/%.png: experiments/%.present.py $(BUILD_DIR)/results/%.npz
	.venv/bin/python $<

LATEX_CMD=pdflatex -output-directory=$(BUILD_DIR)/latextmp
build/pdf/wiggle.pdf: paper.tex $(FIGURES) | $(BUILD_DIR)
	$(LATEX_CMD) -jobname=$(shell basename $@ .pdf) -draftmode $<
	$(LATEX_CMD) -jobname=$(shell basename $@ .pdf) $<
	mv $(BUILD_DIR)/latextmp/$(shell basename $@) $(shell dirname $@)

.PHONY: clean freeze
clean:
	rm -rf $(BUILD_DIR)
	rm -rf experiments/__pycache__
freeze:
	.venv/bin/pip freeze > requirements.txt