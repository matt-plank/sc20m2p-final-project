import argparse

from bing_image_downloader import downloader


def main():
    # Parse the command line arguments to get the dataset name and the number of images to download
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--search-term", type=str, required=True)
    args = parser.parse_args()

    # Download images with bing
    downloader.download(
        args.search_term,
        limit=args.num_images,
        output_dir=args.output_path,
        adult_filter_off=True,
        force_replace=False,
        timeout=3,
    )


if __name__ == "__main__":
    main()
