import argparse
import json
import logging
import sys

# Helper rol function (rotate left)
def rol(val, r_bits, max_bits=64):
    return ((val << r_bits) & (2**max_bits - 1)) | (val >> (max_bits - r_bits))

def signal_spiral_encrypt(block, key, rounds=16, modulus=(2**64 - 59)):
    b = block
    subkeys = [(key >> (i * 8)) & 0xFFFFFFFFFFFFFFFF for i in range(rounds)]
    history = []
    waveform_data = []

    for i in range(rounds):
        k = subkeys[i % len(subkeys)]
        even = (b % 2 == 0)
        history.append((b, even, k))

        # Capture least significant byte for waveform visualization
        waveform_data.append(b & 0xFF)

        if even:
            b = ((b ^ k) >> 1) + k
        else:
            b = ((3 * b + k) ^ rol(k, b % 32)) % modulus

    return b, history, waveform_data

def signal_spiral_decrypt(ciphertext, key, history, modulus=(2**64 - 59)):
    b = ciphertext
    for original, even, k in reversed(history):
        if even:
            b = ((b - k) << 1) ^ k
        else:
            b = ((b ^ rol(k, original % 32)) - k) // 3
    return b

DEFAULT_KEY = 0xDEADBEEFCAFEBABE1234567890ABCDEF
DEFAULT_BLOCK = 0x1122334455667788

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        filename='cipher.log',
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=level,
    )

def save_history(history, path):
    """Save encryption history to JSON file."""
    with open(path, 'w') as f:
        json.dump(history, f)

def load_history(path):
    """Load encryption history from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def export_result(result, path):
    """Export result dictionary to JSON file."""
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)

def read_input(source, arg_name):
    """Read integer input from a file or parse directly."""
    try:
        # Try to open as file
        with open(source, 'r') as f:
            data = f.read().strip()
        value = int(data, 0)
        logging.debug(f"Read {arg_name} from file '{source}': {value}")
        return value
    except FileNotFoundError:
        # Not a file, try parse as int directly
        try:
            value = int(source, 0)
            logging.debug(f"Read {arg_name} from direct input: {value}")
            return value
        except Exception as e:
            logging.error(f"Failed to parse {arg_name}: {e}")
            raise ValueError(f"Invalid {arg_name} input '{source}'. Must be an integer or file containing integer.")

def validate_positive_int(value, name):
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")

def main():
    """
    CLI for Collatz Chaos Cipher encryption and decryption.
    Supports encryption, decryption, file I/O, logging, and export.
    """
    parser = argparse.ArgumentParser(description="Collatz Chaos Cipher CLI Tool")
    parser.add_argument("--block", help="Plaintext block (int or path to file). Default: 0x1122334455667788")
    parser.add_argument("--ciphertext", help="Ciphertext (int or path to file) to decrypt")
    parser.add_argument("--key", help="Encryption key (int or path to file). Default: 0xDEADBEEFCAFEBABE1234567890ABCDEF")
    parser.add_argument("--encrypt", action="store_true", help="Perform encryption")
    parser.add_argument("--decrypt", action="store_true", help="Perform decryption")
    parser.add_argument("--rounds", type=int, default=16, help="Number of encryption rounds (default: 16)")
    parser.add_argument("--modulus", type=lambda x: int(x, 0), default=(2**64 - 59), help="Modulus (default: 2^64 - 59)")
    parser.add_argument("--save-history", type=str, help="Path to save encryption history (JSON)")
    parser.add_argument("--load-history", type=str, help="Path to load encryption history (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Display round-by-round encryption info")
    parser.add_argument("--export", type=str, help="Path to export final result (JSON)")
    parser.add_argument("--output", type=str, help="Write ciphertext or plaintext result to file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to cipher.log")
    parser.add_argument("--test", action="store_true", help="Run unit tests (requires pytest)")

    args = parser.parse_args()
    setup_logging(args.debug)
    logging.info("Started cipher CLI")

    if args.test:
        import subprocess
        logging.info("Running unit tests")
        try:
            subprocess.run(["pytest", "test_vectors.py"], check=True)
        except Exception as e:
            logging.error(f"Unit tests failed: {e}")
            print(f"Unit tests failed: {e}")
        return

    # Load key and block/ciphertext with defaults and validation
    try:
        key = read_input(args.key, "key") if args.key else DEFAULT_KEY
        validate_positive_int(key, "Key")
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.encrypt:
        if args.block is None:
            block = DEFAULT_BLOCK
            print(f"No --block specified; using default plaintext block: 0x{block:X}")
        else:
            try:
                block = read_input(args.block, "block")
                validate_positive_int(block, "Block")
            except ValueError as e:
                print(f"Error: {e}")
                return

        # Encrypt
        logging.info(f"Encrypting block=0x{block:X} with key=0x{key:X}")
        ciphertext, history, waveform_data = signal_spiral_encrypt(block, key, rounds=args.rounds, modulus=args.modulus)
        print(f"Encrypted: 0x{ciphertext:X}")

        if args.verbose:
            for i, (b, even, k) in enumerate(history):
                state = "Even" if even else "Odd"
                print(f"Round {i+1:02}: b = {b}, state = {state}, key = {k}")

        if args.save_history:
            save_history(history, args.save_history)
            print(f"Saved encryption history to {args.save_history}")

        if args.export:
            export_result({"ciphertext": ciphertext, "history": history, "waveform": waveform_data}, args.export)
            print(f"Exported result JSON to {args.export}")

        if args.output:
            with open(args.output, "w") as f:
                f.write(f"{ciphertext}\n")
            print(f"Ciphertext written to {args.output}")

    elif args.decrypt:
        if args.ciphertext is None or args.load_history is None:
            print("Error: --ciphertext and --load-history are required for decryption.")
            return

        try:
            ciphertext = read_input(args.ciphertext, "ciphertext")
            validate_positive_int(ciphertext, "Ciphertext")
        except ValueError as e:
            print(f"Error: {e}")
            return

        try:
            history = load_history(args.load_history)
        except Exception as e:
            print(f"Error loading history file: {e}")
            return

        logging.info(f"Decrypting ciphertext=0x{ciphertext:X} with key=0x{key:X}")
        plaintext = signal_spiral_decrypt(ciphertext, key, history, modulus=args.modulus)
        print(f"Decrypted: 0x{plaintext:X}")

        if args.export:
            export_result({"plaintext": plaintext}, args.export)
            print(f"Exported plaintext JSON to {args.export}")

        if args.output:
            with open(args.output, "w") as f:
                f.write(f"{plaintext}\n")
            print(f"Plaintext written to {args.output}")

    else:
        print("Please specify --encrypt or --decrypt")
        logging.warning("No operation specified (encrypt or decrypt)")

if __name__ == "__main__":
    main()
