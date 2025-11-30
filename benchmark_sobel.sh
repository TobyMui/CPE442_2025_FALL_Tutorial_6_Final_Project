#!/bin/bash

EXEC="./optimized_final"      # <-- CHANGE THIS
INPUT="romania.mov"   # <-- CHANGE THIS
RUNS=5

OUT_DIR="benchmark_results"
mkdir -p "$OUT_DIR"

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

    LOG_FILE="${OUT_DIR}/run_${i}.log"

    # Capture output + time
    {
        /usr/bin/time -f "WALL_TIME %e" $EXEC "$INPUT"
    } &> "$LOG_FILE"

    # --- Extract wall clock time ---
    wall=$(grep "WALL_TIME" "$LOG_FILE" | awk '{print $2}')
    # requires 'bc' installed
    total_wall=$(echo "$total_wall + $wall" | bc)

    # --- Extract Greyscale metrics ---
    grey_line=$(grep "Greyscale" "$LOG_FILE" || true)

    if [[ -n "$grey_line" ]]; then
        grey_cycles=$(echo "$grey_line" | awk '{print $5}')
        grey_l1=$(echo "$grey_line" | awk '{print $10}')
        grey_l2=$(echo "$grey_line" | awk '{print $15}')

        total_grey_cycles=$((total_grey_cycles + grey_cycles))
        total_grey_l1=$((total_grey_l1 + grey_l1))
        total_grey_l2=$((total_grey_l2 + grey_l2))
    else
        echo "Warning: no Greyscale line found in $LOG_FILE"
    fi

    # --- Extract Sobel metrics ---
    sobel_line=$(grep "^Sobel" "$LOG_FILE" || true)

    if [[ -n "$sobel_line" ]]; then
        sobel_cycles=$(echo "$sobel_line" | awk '{print $5}')
        sobel_l1=$(echo "$sobel_line" | awk '{print $10}')
        sobel_l2=$(echo "$sobel_line" | awk '{print $15}')

        total_sobel_cycles=$((total_sobel_cycles + sobel_cycles))
        total_sobel_l1=$((total_sobel_l1 + sobel_l1))
        total_sobel_l2=$((total_sobel_l2 + sobel_l2))
    else
        echo "Warning: no Sobel line found in $LOG_FILE"
    fi
done

echo ""
echo "======================================="
echo "              AVERAGES"
echo "======================================="

avg_wall=$(echo "scale=6; $total_wall / $RUNS" | bc)
avg_grey_cycles=$((total_grey_cycles / RUNS))
avg_grey_l1=$((total_grey_l1 / RUNS))
avg_grey_l2=$((total_grey_l2 / RUNS))
avg_sobel_cycles=$((total_sobel_cycles / RUNS))
avg_sobel_l1=$((total_sobel_l1 / RUNS))
avg_sobel_l2=$((total_sobel_l2 / RUNS))

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
