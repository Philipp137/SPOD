# Predefined variables
ZENODO_ID := 13355796
FILE1 := 2cylinder
FILE2 := Wildlandfire_1d
FILE3 := Wildlandfire_2d

EXAMPLE1 := synthetic_examples_1D.py
EXAMPLE2 := redundant_frame.py
EXAMPLE3 := timing.py
EXAMPLE4 := wildlandfire_1D.py
EXAMPLE5 := wildlandfire_2D.py
EXAMPLE6 := 2_cyls_example.py

# Default target
.PHONY: all
all: download run

# Step 1: Create the "data" directory
data:
	mkdir -p data

# Step 2: Change into the "data" directory and download files using zenodo_get
download: data
	@cd data && \
	if ! command -v zenodo_get >/dev/null 2>&1; then \
		echo "Please install 'zenodo_get' by running: pip3 install zenodo_get"; \
		exit 1; \
	fi && \
	zenodo_get $(ZENODO_ID) && \
	unzip -d $(FILE1) $(FILE1).zip && \
	unzip -d $(FILE2) $(FILE2).zip && \
	unzip -d $(FILE3) $(FILE3).zip && \
	touch downloaded

# Step 3: Run all Python examples
.PHONY: run
run: all
	python3 $(EXAMPLE1)
	python3 $(EXAMPLE2)
	python3 $(EXAMPLE3)
	python3 $(EXAMPLE4)
	python3 $(EXAMPLE5)
	python3 $(EXAMPLE6)


# Cleanup
.PHONY: clean
clean:
	rm -rf data images
