from cipher_core import signal_spiral_encrypt, signal_spiral_decrypt

def main():
    key = 0xDEADBEEFCAFEBABE1234567890ABCDEF
    plaintext = 0x1122334455667788

    ciphertext, history = signal_spiral_encrypt(plaintext, key)
    print(f"Ciphertext: 0x{ciphertext:X}")

    decrypted = signal_spiral_decrypt(ciphertext, key, history)
    print(f"Decrypted: 0x{decrypted:X}")

if __name__ == "__main__":
    main()
