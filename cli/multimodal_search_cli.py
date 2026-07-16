import argparse
from lib.multimodal_search import verify_image_embedding, search_by_image

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    vie_parser = subparsers.add_parser(
        "verify_image_embedding", help="Create CLIP embedding and print"
    )
    vie_parser.add_argument("image_path", type=str, help="path to the image to embed.")

    img_search_parser = subparsers.add_parser(
        "image_search", help="Search movies by image."
    )
    img_search_parser.add_argument("image_path", type=str, help="path to the source image.")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            if args.image_path != "":
                verify_image_embedding(args.image_path)

        case "image_search":
            if args.image_path != "":
                results = search_by_image(args.image_path)
                for i,r in enumerate(results):
                    print(f"{i}. {r['title']} (similarity: {r['score']:.3f})\n{r['description'][:100]}\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
