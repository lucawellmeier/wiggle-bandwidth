BUILD_DIR=build
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/latextmp
	mkdir -p $(BUILD_DIR)/pdf

LATEX_CMD=pdflatex -output-directory=$(BUILD_DIR)/latextmp
build/pdf/wiggle.pdf: paper.tex | $(BUILD_DIR)
	$(LATEX_CMD) -jobname=$(shell basename $@ .pdf) -draftmode $^
	$(LATEX_CMD) -jobname=$(shell basename $@ .pdf) $^
	mv $(BUILD_DIR)/latextmp/$(shell basename $@) $(shell dirname $@)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)