# Predefined variables
ZENODO_ID := 3909642
FILE1 := BlaSU20_code_nonperiodic
FILE2 := BlaSU20_code_periodic
FILE3 := file3

EXAMPLE1 := traveling_wave_1D.py
EXAMPLE2 := timing.py
EXAMPLE3 := wildlandfire_1D.py
EXAMPLE3 := wildlandfire_2D.py
EXAMPLE4 := 2_cyls_example.py

# Default target
.PHONY: all
all: data/downloaded run

# Step 1: Create the "data" directory
data:
	mkdir -p data

# Step 2: Change into the "data" directory and download files using zenodo_get
data/downloaded: data
	@cd data && \
	if ! command -v zenodo_get >/dev/null 2>&1; then \
		echo "Please install 'zenodo_get' by running: pip3 install zenodo_get"; \
		exit 1; \
	fi && \
	zenodo_get $(ZENODO_ID) && \
	unzip -d $(FILE1) $(FILE1).zip && \
	unzip -d $(FILE2) $(FILE2).zip && \
	touch downloaded

# Step 3: Run all Python examples
.PHONY: run
run: all
	python3 $(EXAMPLE1)
	python3 $(EXAMPLE2)
	python3 $(EXAMPLE3)

# Cleanup
.PHONY: clean
clean:
	rm -rf data images
