# Makefile for EZ Bootstrap (EZB) C implementation using GSL

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lgsl -lgslcblas -lm

# Common source
COMMON_SRC = ezb_common.c
COMMON_OBJ = ezb_common.o

# Targets
TARGETS = ezb_single ezb_design_matrix test_ezb

all: $(TARGETS)

ezb_single: ezb_single.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COMMON_OBJ) $(LDFLAGS)

ezb_design_matrix: ezb_design_matrix.c ezb_design_matrix_impl.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ ezb_design_matrix.c ezb_design_matrix_impl.c $(COMMON_OBJ) $(LDFLAGS)

test_ezb: test_ezb.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(COMMON_OBJ): $(COMMON_SRC) ezb_common.h
	$(CC) $(CFLAGS) -c -o $@ $(COMMON_SRC)

clean:
	rm -f $(TARGETS) $(COMMON_OBJ) *.o

.PHONY: all clean

