CXX      = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Wno-unused-variable
# Search compat shims first, then the actual Allen headers
ALLEN_HEADERS = ../../device/kalman/ParKalman/include
INCLUDES = -I. -I$(ALLEN_HEADERS)

TARGET   = run_extrapolators
SRC      = main.cpp

all: $(TARGET)

$(TARGET): $(SRC) cuda_compat.h MagneticField.cuh BackendCommon.h FloatOperations.cuh MLPExtrapolator.cuh
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) trajectories.csv kalman_results.csv kalman_hits.csv \
	      kalman_results_mlp.csv kalman_hits_mlp.csv

.PHONY: all run clean
