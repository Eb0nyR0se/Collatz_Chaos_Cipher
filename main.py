# File: main.py

import matplotlib
matplotlib.use("TkAgg")  # Force GUI backend (change to "Qt5Agg" if preferred)
import logging
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from cipher_core import (derive_key_from_ec, handle_encrypt,
                         handle_decrypt, handle_visualize, bifurcation_diagram,
                         interactive_bifurcation)


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        filename='main.log',
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=level,
    )


def interactive_menu():
    print("""
-------------------------------------------------------
    Welcome to the Collatz Chaos Cipher Toolkit 🌀
-------------------------------------------------------
You are using a 256-bit encryption algorithm fused with
non-integer Collatz dynamics. Encryption is irreversible,
chaotic, and visualizable in real time.

Available Commands:
  1. Encrypt a float block
  2. Decrypt (placeholder)
  3. Visualize 2D Heatmap
  4. Visualize 3D Encryption Surface
  5. Visualize Bifurcation Diagram
  6. Export Encryption Data to CSV
  7. Visualize Phase Space Plot
  8. Exit
-------------------------------------------------------
""")
    while True:
        choice = input("Select an option (1–8): ").strip()
        if choice == "8":
            print("Goodbye!")
            break
        if choice not in {"1", "2", "3", "4", "5", "6", "7"}:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, 7, or 8.")
            continue

        try:
            block = float(input("Enter block (float): "))
            use_ec = input("Use EC-derived key? (y/n): ").lower().startswith("y")
            if use_ec:
                key = derive_key_from_ec()
                print(f"Derived EC key: {key}")
            else:
                key = float(input("Enter key (float): "))
            rounds = int(input("Enter number of rounds (default 100): ") or "100")
        except Exception as e:
            print(f"Input error: {e}")
            continue

        if choice == "1":
            handle_encrypt(block, key, rounds)
        elif choice == "2":
            handle_decrypt(block, key, rounds)
        elif choice == "3":
            from Heatmap2D import run_heatmap_gui
            print("Launching 2D Heatmap visualization...")
            run_heatmap_gui()
        elif choice == "4":
            from Encryption_surface3D import generate_surface_data, plot_surface
            print("Launching 3D Encryption Surface visualization...")
            key_start = key
            key_end = key + 1
            steps = 50
            keys, val, wave, diff, ent = generate_surface_data(block, key_start, key_end, steps, rounds)
            plot_surface(keys, val, wave, diff, ent, rounds, interactive=True)
        elif choice == "5":
            print("Launching Bifurcation Diagram visualization...")
            interactive_bifurcation()
        elif choice == "6":
            print("Exporting encryption data to CSV...")
            filename = input("Enter CSV filename (default: encryption_data.csv): ").strip() or "encryption_data.csv"
            handle_visualize(block, key, rounds, save=True, filename=filename, export_csv=True, csv_filename=filename)
            print(f"Encryption data exported to {filename}")
        elif choice == "7":
            from phase_space import phase_space_plot
            print("Launching Phase Space Plot visualization...")
            phase_space_plot(block, key, rounds)


def main():

    setup_logging()

    parser = argparse.ArgumentParser(description="Collatz Chaos Cipher CLI with Visualization")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Encrypt/Decrypt/Visualize subcommands
    for cmd in ["encrypt", "decrypt", "visualize"]:
        p = subparsers.add_parser(cmd)
        p.add_argument("--block", type=float, required=True)
        p.add_argument("--key", type=float)
        p.add_argument("--use-ec", action="store_true", help="Use elliptic curve derived key")
        p.add_argument("--rounds", type=int, default=100)
        p.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Plot subcommand for 3D surface visualization
    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--block", type=float, required=True)
    plot_parser.add_argument("--key-start", type=float, required=True)
    plot_parser.add_argument("--key-end", type=float, required=True)
    plot_parser.add_argument("--steps", type=int, default=100)
    plot_parser.add_argument("--rounds", type=int, default=100)
    plot_parser.add_argument("--color-by", choices=["waveform", "diffusion", "entropy"], default="waveform")
    plot_parser.add_argument("--interactive", action="store_true")

    # Heatmap subcommand for 2D waveform heatmaps
    heatmap_parser = subparsers.add_parser("heatmap")
    heatmap_parser.add_argument("--block", type=float, required=True)
    heatmap_parser.add_argument("--key", type=float, required=True)
    heatmap_parser.add_argument("--rounds", type=int, default=100)
    heatmap_parser.add_argument("--mode", choices=["waveform", "keysweep"], default="waveform")
    heatmap_parser.add_argument("--colormap", type=str, default="inferno")
    heatmap_parser.add_argument("--save", type=str, default=None)

    # Bifurcation diagram subcommand
    bif_parser = subparsers.add_parser("bifurcation", help="Generate bifurcation diagram")
    bif_parser.add_argument("--block", type=float, required=True)
    bif_parser.add_argument("--key-start", type=float, required=True)
    bif_parser.add_argument("--key-end", type=float, required=True)
    bif_parser.add_argument("--steps", type=int, default=500)
    bif_parser.add_argument("--rounds", type=int, default=100)

    # Phase space plot subcommand (NEW option)
    phase_parser = subparsers.add_parser("phase_space", help="Generate phase space plot")
    phase_parser.add_argument("--block", type=float, required=True)
    phase_parser.add_argument("--key", type=float, required=True)
    phase_parser.add_argument("--rounds", type=int, default=100)

    # If no args, run interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        return

    args = parser.parse_args()

    if getattr(args, "debug", False):
        setup_logging(debug=True)

    # Handle key derivation
    key = None
    if hasattr(args, "use_ec") and args.use_ec:
        key = derive_key_from_ec()
    elif hasattr(args, "key"):
        key = args.key

    # Dispatch commands
    if args.command in {"encrypt", "decrypt", "visualize"}:
        if key is None:
            print("Please provide --key or use --use-ec")
            return
        if args.command == "encrypt":
            handle_encrypt(args.block, key, args.rounds)
        elif args.command == "decrypt":
            handle_decrypt(args.block, key, args.rounds)
        elif args.command == "visualize":
            handle_visualize(args.block, key, args.rounds)

    elif args.command == "plot":
        from Encryption_surface3D import generate_surface_data, plot_surface
        keys, val, wave, diff, ent = generate_surface_data(
            args.block, args.key_start, args.key_end, args.steps, args.rounds
        )
        plot_surface(
            keys, val, wave, diff, ent, args.rounds,
            color_by=args.color_by,
            interactive=args.interactive
        )

    elif args.command == "heatmap":
        from Heatmap2D import interactive_waveform_heatmap, interactive_keysweep_heatmap

        if args.mode == "waveform":
            fig, ax = interactive_waveform_heatmap(args.block, args.key, max_rounds=args.rounds, cmap=args.colormap)
        else:
            fig, ax = interactive_keysweep_heatmap(args.block, args.key, initial_key_range=0.1 * args.key,
                                                  steps=100, max_rounds=args.rounds, cmap=args.colormap)

        if args.save:
            fig.savefig(args.save)
            print(f"Saved heatmap to {args.save}")

        import matplotlib.pyplot as plt
        plt.show()

    elif args.command == "bifurcation":
        keys, values = bifurcation_diagram(args.block, args.key_start, args.key_end, args.steps, args.rounds)
        # Add plot or save file code here if desired

    elif args.command == "phase_space":
        from phase_space import phase_space_plot
        if key is None:
            print("Please provide --key or use --use-ec")
            return
        phase_space_plot(args.block, key, args.rounds)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
