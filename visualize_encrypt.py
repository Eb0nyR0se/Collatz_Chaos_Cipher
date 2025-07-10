import matplotlib.pyplot as plt
from cipher import signal_spiral_encrypt

def visualize_encryption(block, key, rounds=16, save=False):
    ciphertext, history = signal_spiral_encrypt(block, key, rounds=rounds)

    values = [h[0] for h in history]
    colors = ['blue' if h[1] else 'red' for h in history]  # even = blue, odd = red
    steps = list(range(1, len(values) + 1))

    plt.figure(figsize=(10, 6))
    plt.title("Collatz Chaos Cipher: Encryption Path")
    plt.xlabel("Round")
    plt.ylabel("Block Value")
    plt.grid(True)

    plt.plot(steps, values, '-o', color='gray', alpha=0.5)
    for x, y, c in zip(steps, values, colors):
        plt.scatter(x, y, color=c, s=100)

    blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Even', markerfacecolor='blue', markersize=10)
    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Odd', markerfacecolor='red', markersize=10)
    plt.legend(handles=[blue_patch, red_patch])

    plt.tight_layout()

    if save:
        plt.savefig("encryption_visual.png")
        print("Saved plot as encryption_visual.png")
    else:
        plt.show()

if __name__ == "__main__":
    key = 0xDEADBEEFCAFEBABE1234567890ABCDEF
    block = 0x1122334455667788
    visualize_encryption(block, key, save=False)
