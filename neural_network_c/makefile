CC = gcc
CFLAGS = -Iinclude -Wall -Wextra -O2 -lm
SRC_DIR = src
TEST_DIR = test

SRC = $(wildcard $(SRC_DIR)/*.c)
TEST_SRC = $(wildcard $(TEST_DIR)/*.c)

all: nn test

nn: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^
	
test: $(filter-out $(SRC_DIR)/main.c, $(SRC)) $(filter-out $(TEST_DIR)/test_*.c, $(TEST_SRC)) $(TEST_DIR)/main_test.c
	$(CC) $(CFLAGS) -o test_nn $^
       
clean:
	rm -f nn test_nn *.o
