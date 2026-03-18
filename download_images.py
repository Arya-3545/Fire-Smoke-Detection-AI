from bing_image_downloader import downloader

# Download fire images
downloader.download(
    query="fire flames",
    limit=500,
    output_dir="dataset",
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)

# Download smoke images
downloader.download(
    query="smoke",
    limit=500,
    output_dir="dataset",
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)
