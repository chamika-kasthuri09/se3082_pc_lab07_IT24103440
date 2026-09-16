CC = mpicc
CFLAGS = -Wall -Wextra -std=c99
TARGETS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

all: $(TARGETS)

sum_bcast: sum_bcast.c
	$(CC) $(CFLAGS) -o $@ $<

sum_scatter: sum_scatter.c
	$(CC) $(CFLAGS) -o $@ $<

sum_gather: sum_gather.c
	$(CC) $(CFLAGS) -o $@ $<

sum_reduce: sum_reduce.c
	$(CC) $(CFLAGS) -o $@ $<

sum_allreduce: sum_allreduce.c
	$(CC) $(CFLAGS) -o $@ $<

sum_scan: sum_scan.c
	$(CC) $(CFLAGS) -o $@ $<

run: all
	@echo "Running with 4 processes:"
	@for prog in $(TARGETS); do \
		echo "=== $$prog ==="; \
		mpirun -np 4 ./$$prog; \
		echo ""; \
	done

clean:
	rm -f $(TARGETS) *.o