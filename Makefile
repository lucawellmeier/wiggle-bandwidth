BUILD_DIR=build
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/latextmp
	mkdir -p $(BUILD_DIR)/pdf
	mkdir -p $(BUILD_DIR)/results
	mkdir -p $(BUILD_DIR)/figures

EXPERIMENTS = $(shell find experiments -name "exp_*.py")
RESULTS = $(patsubst experiments/exp_%.py,$(BUILD_DIR)/results/%.npz,$(EXPERIMENTS))
FIGURES = $(patsubst experiments/exp_%.py,$(BUILD_DIR)/figures/%.png,$(EXPERIMENTS))
.PRECIOUS: $(RESULTS)

$(BUILD_DIR)/results/%.npz: experiments/exp_%.py | $(BUILD_DIR)
	python -c "from experiments.exp_$* import compute; compute()"
$(BUILD_DIR)/figures/%.png: $(BUILD_DIR)/results/%.npz
	python -c "from experiments.exp_$* import figure; figure(show=False)"

LATEX_CMD=pdflatex -output-directory=$(BUILD_DIR)/latextmp
build/pdf/wiggle.pdf: paper.tex $(FIGURES) | $(BUILD_DIR)
	$(LATEX_CMD) -jobname=$(shell basename $@ .pdf) -draftmode $<
	$(LATEX_CMD) -jobname=$(shell basename $@ .pdf) $<
	mv $(BUILD_DIR)/latextmp/$(shell basename $@) $(shell dirname $@)

.PHONY: clean freeze
clean:
	rm -rf $(BUILD_DIR)
freeze:
	.venv/bin/pip freeze > requirements.txt