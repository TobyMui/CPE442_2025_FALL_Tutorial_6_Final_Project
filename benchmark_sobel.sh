#!/bin/bash

EXEC="./sobel_PAPI"      # <-- CHANGE IF NEEDED
INPUT="romania.mov"           # <-- CHANGE IF NEEDED
RUNS=5

OUT_FILE="baseline_3.9Ghz"

# Accumulators
total_wall=0
total_grey_cycles=0
total_grey_l1=0
total_grey_l2=0
total_sobel_cycles=0
total_sobel_l1=0
total_sobel_l2=0

echo "======================================="
echo " Benchmarking $EXEC for $RUNS runs"
echo "======================================="

for i in $(seq 1 $RUNS); do
    echo "Running test $i ..."

    # Capture output of the program into a variable
    OUTPUT=$(/usr/bin/time -f "WALL_TIME %e" $EXEC "$INPUT" 2>&1)

    # --- Extract wall time ---
    wall=$(echo "$OUTPUT" | grep "WALL_TIME" | awk '{print $2}')
    total_wall=$(echo "$total_wall + $wall" | bc)

    # --- Extract grep lines from OUTPUT ---
    grey_line=$(echo "$OUTPUT" | grep "Greyscale" || true)
    sobel_line=$(echo "$OUTPUT" | grep "^Sobel" || true)

    # --- Parse Greyscale metrics ---
    if [[ -n "$grey_line" ]]; then
        grey_cycles=$(echo "$grey_line" | awk '{print $5}')
        grey_l1=$(echo "$grey_line" | awk '{print $10}')
        grey_l2=$(echo "$grey_line" | awk '{print $15}')

        total_grey_cycles=$((total_grey_cycles + grey_cycles))
        total_grey_l1=$((total_grey_l1 + grey_l1))
        total_grey_l2=$((total_grey_l2 + grey_l2))
    fi

    # --- Parse Sobel metrics ---
    if [[ -n "$sobel_line" ]]; then
        sobel_cycles=$(echo "$sobel_line" | awk '{print $5}')
        sobel_l1=$(echo "$sobel_line" | awk '{print $10}')
        sobel_l2=$(echo "$sobel_line" | awk '{print $15}')

        total_sobel_cycles=$((total_sobel_cycles + sobel_cycles))
        total_sobel_l1=$((total_sobel_l1 + sobel_l1))
        total_sobel_l2=$((total_sobel_l2 + sobel_l2))
    fi
done

# ---- Compute Averages ----
avg_wall=$(echo "scale=6; $total_wall / $RUNS" | bc)
avg_grey_cycles=$((total_grey_cycles / RUNS))
avg_grey_l1=$((total_grey_l1 / RUNS))
avg_grey_l2=$((total_grey_l2 / RUNS))
avg_sobel_cycles=$((total_sobel_cycles / RUNS))
avg_sobel_l1=$((total_sobel_l1 / RUNS))
avg_sobel_l2=$((total_sobel_l2 / RUNS))

# ---- Write to output file ----
{
    echo "======================================="
    echo " Benchmark Summary"
    echo " Executable: $EXEC"
    echo " Runs: $RUNS"
    echo " Input: $INPUT"
    echo "======================================="
    echo "Average Wall Time:     $avg_wall seconds"
    echo ""
    echo "Greyscale Averages:"
    echo "  Cycles:              $avg_grey_cycles"
    echo "  L1 Cache Miss:       $avg_grey_l1"
    echo "  L2 Cache Miss:       $avg_grey_l2"
    echo ""
    echo "Sobel Averages:"
    echo "  Cycles:              $avg_sobel_cycles"
    echo "  L1 Cache Miss:       $avg_sobel_l1"
    echo "  L2 Cache Miss:       $avg_sobel_l2"
    echo "======================================="
} > "$OUT_FILE"

# Print results to terminal too
cat "$OUT_FILE"

echo ""
echo "Results saved to: $OUT_FILE"
