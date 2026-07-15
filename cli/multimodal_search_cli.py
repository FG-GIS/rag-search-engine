import argparse
from lib.multimodal_search import verify_image_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    vie_parser = subparsers.add_parser(
        "verify_image_embedding", help="Create CLIP embedding and print"
    )
    vie_parser.add_argument("image_path", type=str, help="path to the image to embed.")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            if args.image_path != "":
                verify_image_embedding(args.image_path)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
