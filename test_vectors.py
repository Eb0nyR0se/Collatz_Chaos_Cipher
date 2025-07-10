# File: test_vectors.py

#File: test_vectors.py

from main import signal_spiral_encrypt, signal_spiral_decrypt


def run_test_vectors():
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

    while True:
        use_custom = input("Run custom test? (y/n, Enter to run defaults): ").strip().lower()
        if use_custom == 'y':
            pt = input_hex("Enter plaintext (hex): ")
            key = input_hex("Enter key (hex): ")
            rounds = input_rounds("Enter number of rounds (default 16): ")
            test_cases = [{"plaintext": pt, "key": key, "rounds": rounds}]
        else:
            if use_custom == 'n' or use_custom == '':
                # run defaults
                pass
            else:
                print("Invalid input. Please enter 'y', 'n', or Enter.")
                continue

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

        if use_custom in ('n', ''):
            break


def input_hex(prompt):
    while True:
        val = input(prompt).strip()
        if val.startswith("0x") or val.startswith("0X"):
            val = val[2:]
        try:
            return int(val, 16)
        except ValueError:
            print("Invalid hex value. Please try again.")


def input_rounds(prompt):
    val = input(prompt).strip()
    if val == '':
        return 16
    try:
        rounds = int(val)
        if rounds <= 0:
            raise ValueError
        return rounds
    except ValueError:
        print("Invalid number of rounds. Using default 16.")
        return 16


if __name__ == "__main__":
    run_test_vectors()

