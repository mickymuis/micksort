# General flags
CC = mpicc
CFLAGS = -Wall -O3 -g -fopenmp -march=native
LDFLAGS = -lm

BIN = build/micksort

OBJS = build/micksort.o
HEADERS = 
SOURCES = src/micksort.c

all:		$(BIN)

$(BIN):		$(OBJS)
		$(CC) $(CFLAGS) $(OBJS) -o $(BIN) $(LDFLAGS)

build/%.o:	src/%.c $(HEADERS)
		mkdir -p build
		$(CC) $(CFLAGS) -c $< -o $@

clean:
		rm -rf build

