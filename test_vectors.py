# File: test_vectors.py


import argparse
from cipher_core import signal_spiral_decrypt, signal_spiral_encrypt


def run_test_vectors(custom_plaintext=None, custom_key=None, custom_rounds=16):
    test_cases = []

    if custom_plaintext is not None and custom_key is not None:
        test_cases.append({
            "plaintext": custom_plaintext,
            "key": custom_key,
            "rounds": custom_rounds
        })
    else:
        test_cases = [
            {
                "plaintext": 0x1122334455,
                "key": 0x4242424242424242,
                "rounds": 16,
            },
            {
                "plaintext": 0x1122334455667788,
                "key": 0x4242424242424242,
                "rounds": 16,
            },
            {
                "plaintext": 0x0000000000000001,
                "key": 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F,
                "rounds": 16,
            },
            {
                "plaintext": 0xFFFFFFFFFFFFFFFF,
                "key": 0x123456789ABCDEF0123456789ABCDEF0,
                "rounds": 16,
            }
        ]

    print("Running Collatz Chaos Cipher Test Vectors...\n")

    for i, case in enumerate(test_cases, 1):
        pt = case["plaintext"]
        key = case["key"]
        rounds = case.get("rounds", 16)

        ct, history, _ = signal_spiral_encrypt(pt, key, rounds=rounds)
        recovered = signal_spiral_decrypt(ct, key, history)

        print(f"Test Case {i}")
        print(f"Plaintext : 0x{pt:016X}")
        print(f"Key       : 0x{key:032X}")
        print(f"Rounds    : {rounds}")
        print(f"Ciphertext: 0x{int(ct):016X}")
        print(f"Decrypted : 0x{int(recovered):016X}")
        print("PASS\n" if int(pt) == int(recovered) else "FAIL\n")


def main():
    parser = argparse.ArgumentParser(description="Run Collatz Chaos Cipher test vectors")
    parser.add_argument("--plaintext", type=lambda x: int(x, 16), help="Custom plaintext in hex (e.g. 0x1122334455)")
    parser.add_argument("--key", type=lambda x: int(x, 16), help="Custom key in hex (e.g. 0x4242424242424242)")
    parser.add_argument("--rounds", type=int, default=16, help="Number of encryption rounds (default: 16)")

    args = parser.parse_args()

    run_test_vectors(custom_plaintext=args.plaintext, custom_key=args.key, custom_rounds=args.rounds)


if __name__ == "__main__":
    main()
