from cipher import signal_spiral_encrypt, signal_spiral_decrypt

def run_test_vectors():
    test_cases = [
        {
            "plaintext": 0x1122334455667788,
            "key": 0xDEADBEEFCAFEBABE1234567890ABCDEF,
        },
        {
            "plaintext": 0x0000000000000001,
            "key": 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F,
        },
        {
            "plaintext": 0xFFFFFFFFFFFFFFFF,
            "key": 0x123456789ABCDEF0123456789ABCDEF0,
        }
    ]

    print("Running Collatz Chaos Cipher Test Vectors...\n")

    for i, case in enumerate(test_cases, 1):
        pt = case["plaintext"]
        key = case["key"]

        ct, history = signal_spiral_encrypt(pt, key)
        recovered = signal_spiral_decrypt(ct, key, history)

        print(f"Test Case {i}")
        print(f"Plaintext : 0x{pt:016X}")
        print(f"Key       : 0x{key:032X}")
        print(f"Ciphertext: 0x{ct:016X}")
        print(f"Decrypted : 0x{recovered:016X}")
        print("PASS\n" if pt == recovered else "FAIL\n")

if __name__ == "__main__":
    run_test_vectors()
