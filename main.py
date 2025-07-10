# File: cipher.py

import argparse
import json
import logging
import math

def fractional_part(x):
    return x - math.floor(x)

def signal_spiral_encrypt(block, key, rounds=16):
    b = float(block)
    k = float(key)
    history = []
    waveform_data = []

    for i in range(rounds):
        frac = fractional_part(b)
        # Define "even" as fractional part < 0.5 (heuristic)
        even = frac < 0.5

        # Record state for history and waveform visualization
        history.append((b, even, k))
        waveform_data.append(int(frac * 255))

        if even:
            # Non-integer Collatz "even" step with key mixing
            b = (b / 2.0) + k
        else:
            # Non-integer Collatz "odd" step with key mixing
            b = (3 * b + k) / 2.0

    return b, history, waveform_data

def signal_spiral_decrypt(ciphertext, _key, history):
    b = float(ciphertext)
    # k = float(_key)  # Not needed, remove this line

    # Reverse through history
    for original, even, key_val in reversed(history):
        if even:
            # Inverse of encryption "even" step
            b = 2 * (b - key_val)
        else:
            # Inverse of encryption "odd" step
            b = (2 * b - key_val) / 3

    return b

default_key = 0x4242424242424242
default_block = 0x1122334455667788

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
    # Convert floats to strings for JSON serialization
    serializable_history = [
        (str(b), even, str(k)) for b, even, k in history
    ]
    json_str = json.dumps(serializable_history)
    with open(path, 'w') as f:
        f.write(json_str)

def load_history(path):
    """Load encryption history from JSON file."""
    raw_history = json.load(open(path, 'r'))
    # Convert strings back to floats
    history = [
        (float(b), even, float(k)) for b, even, k in raw_history
    ]
    return history

def export_result(result, path):
    """Export result dictionary to JSON file."""
    json_str = json.dumps(result, indent=2)
    with open(path, 'w') as f:
        f.write(json_str)

def read_input(source, arg_name):
    """Read float input from file or parse directly."""
    try:
        with open(source, 'r') as f:
            data = f.read().strip()
        value = float(data)
        logging.debug(f"Read {arg_name} from file '{source}': {value}")
        return value
    except FileNotFoundError:
        try:
            value = float(source)
            logging.debug(f"Read {arg_name} from direct input: {value}")
            return value
        except Exception as e:
            logging.error(f"Failed to parse {arg_name}: {e}")
            raise ValueError(f"Invalid {arg_name} input '{source}'. Must be a float or file containing a float.")

def validate_positive_float(value, name):
    if value < 0.0:
        raise ValueError(f"{name} must be a non-negative float.")

def main():
    parser = argparse.ArgumentParser(description="Collatz Chaos Cipher CLI Tool (float version)")
    parser.add_argument("--block", help="Plaintext block (float or path to file). Default: 12345.6789")
    parser.add_argument("--ciphertext", help="Ciphertext (float or path to file) to decrypt")
    parser.add_argument("--key", help="Encryption key (float or path to file). Default: 98765.4321")
    parser.add_argument("--encrypt", action="store_true", help="Perform encryption")
    parser.add_argument("--decrypt", action="store_true", help="Perform decryption")
    parser.add_argument("--rounds", type=int, default=16, help="Number of encryption rounds (default: 16)")
    parser.add_argument("--save-history", type=str, help="Path to save encryption history (JSON)")
    parser.add_argument("--load-history", type=str, help="Path to load encryption history (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Display round-by-round encryption info")
    parser.add_argument("--export", type=str, help="Path to export final result (JSON)")
    parser.add_argument("--output", type=str, help="Write ciphertext or plaintext result to file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to cipher.log")

    args = parser.parse_args()
    setup_logging(args.debug)
    logging.info("Started float cipher CLI")

    default_block_float = 12345.6789
    default_key_float = 98765.4321

    try:
        key = read_input(args.key, "key") if args.key else default_key_float
        validate_positive_float(key, "Key")
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.encrypt:
        if args.block is None:
            block = default_block_float
            print(f"No --block specified; using default plaintext block: {block}")
        else:
            try:
                block = read_input(args.block, "block")
                validate_positive_float(block, "Block")
            except ValueError as e:
                print(f"Error: {e}")
                return

        logging.info(f"Encrypting block={block} with key={key}")
        ciphertext, history, waveform_data = signal_spiral_encrypt(block, key, rounds=args.rounds)
        print(f"Encrypted: {ciphertext}")

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
            validate_positive_float(ciphertext, "Ciphertext")
        except ValueError as e:
            print(f"Error: {e}")
            return

        try:
            history = load_history(args.load_history)
        except Exception as e:
            print(f"Error loading history file: {e}")
            return

        logging.info(f"Decrypting ciphertext={ciphertext} with key={key}")
        plaintext = signal_spiral_decrypt(ciphertext, key, history)
        print(f"Decrypted: {plaintext}")

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
