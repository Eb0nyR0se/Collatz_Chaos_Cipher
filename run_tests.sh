#File: run_tests.sh

#!/bin/bash

echo "Running Collatz Chaos Cipher unit tests..."

python -m unittest discover -s . -p "test_vectors.py"

if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
else
    echo "Some tests failed. Please review the output above."
fi
