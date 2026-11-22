.PHONY: all build verify-c-compile rebuild-simd test selftest clean

CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -fPIC -Wall -Wextra

all: build verify-c-compile test selftest

build:
	@test -f libprod_gate_simd.so || $(CC) $(CFLAGS) -shared -o libprod_gate_simd.so prod_gate_native_simd.c -lm

verify-c-compile:
	$(CC) $(CFLAGS) -shared -o /tmp/test_electric_vision_simd.so prod_gate_native_simd.c -lm
	@rm -f /tmp/test_electric_vision_simd.so

rebuild-simd:
	$(CC) $(CFLAGS) -shared -o libprod_gate_simd.so prod_gate_native_simd.c -lm
	@sha256sum libprod_gate_simd.so

test:
	python3 -m unittest discover -v -s . -p "test_*.py"
	python3 test_play_engine_screen.py

selftest:
	python3 electric_vision_cli.py --no-color --plain --mode focus selftest

clean:
	rm -rf __pycache__ .pytest_cache prod_gate_verification_report.json /tmp/test_electric_vision_simd.so
