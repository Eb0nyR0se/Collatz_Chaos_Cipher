# **Collatz Chaos Cipher** 🌀 
#

The Collatz Chaos Cipher is an experimental 256-bit block encryption system that fuses elliptic curve cryptography with a novel non-integer extension of the Collatz Conjecture. Inspired by the famous 3x+1 function, this cipher applies chaotic, nonlinear transformations on floating-point data blocks, producing irreversible, high-entropy pathways that exhibit sensitivity to initial conditions and are visually demonstrable.


Unlike traditional ciphers, the Collatz Chaos Cipher leverages chaotic iteration, signal spiral dynamics, and non-linear key influence to generate avalanche-prone encrypted outputs. The project also includes visualization tools to animate bit diffusion, waveform transformations, and entropy changes during encryption rounds, providing insight into the cipher’s complex behavior.
#
#
**Features:**

256-bit encryption combining elliptic curve cryptography with Collatz-inspired chaos

Non-integer Collatz Conjecture extension, using floating-point arithmetic for dynamic iteration

Signal spiral logic influencing key transformation non-linearly

Reversible block cipher design enabling decryption via inverse chaotic transforms

Real-time visualization of bit diffusion and waveform evolution across rounds

Avalanche effect tests and entropy tracking for cryptanalysis support

Academic documentation and invitation for community cryptanalysis contributions

Branded assets, including chaos spiral logos and SVG/PNG graphics
#
#
**Intended Audience:**


Cryptographers and security researchers interested in experimental cryptography

Mathematicians and chaos theorists studying complex dynamical systems

Signal processing enthusiasts exploring novel encryption concepts

Developers and hobbyists curious about non-traditional block cipher design
#
#
**Important Disclaimer:**

For educational and experimental purposes only! This is NOT secure encryption. It’s a toy example demonstrating how a Collatz-inspired iterative function can be used as a pseudorandom transform.
#
#
**Installation:**

Clone the repository

bash

Copy

Edit

git clone https://github.com/yourusername/collatz-chaos-cipher.git
cd collatz-chaos-cipher

Ensure you have Python 3.x and dependencies installed (e.g., matplotlib, numpy):

bash

Copy

Edit

pip install -r requirements.txt
#
#
**Running Tests:**

Make the test script executable once:

bash

Copy

Edit

chmod +x run_tests.sh
#

**Run all tests:**

bash

Copy

Edit

./run_tests.sh
#
**Or run unit tests manually:**

bash

Copy

Edit

python -m unittest test_vectors.py
#
#
**Visualization:**

The project provides detailed plots visualizing the cipher’s behavior:

Non-integer Collatz trajectories: Evolution of various floating-point starting values under the extended Collatz rule (even: x/2, odd: (3x+1)/2), revealing chaotic decay patterns.

Encryption rounds: Plots showing block values over rounds, with parity indicated, alongside least significant byte waveforms demonstrating bit-level diffusion and nonlinear transformations.
#
#
**Contribution:**

Contributions, cryptanalysis, and academic discussion are welcome. Please submit issues or pull requests via GitHub.
