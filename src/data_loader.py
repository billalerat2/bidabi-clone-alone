"""
OpenFoodFacts product scraper with retry logic, image downloading,
CSV export and category filtering.

Used to collect a dataset of food products for machine learning tasks.
"""

import csv
import time
import requests
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://world.openfoodfacts.org/api/v2/search"
HEADERS = {"User-Agent": "MyAwesomeApp/1.0"}
PRODUCT_FIELDS = (
    "code,product_name,categories_tags,ingredients_text,"
    "image_url,image_front_url,image_small_url,image_thumb_url"
)

TARGET_COUNT = 180
PAGE_SIZE = 100
MAX_PAGES = 50

OUTPUT_ROOT = os.path.join("data", "raw")
IMAGES_ROOT = os.path.join(OUTPUT_ROOT, "images")

CATEGORIES = [
    "sugar",
    "chocolates",
    "breads",
    "cheeses",
]


# --- Session with retry ---
def create_session():
    """
    Creates a requests.Session configured with retry logic.

    Returns
    -------
    requests.Session
        Session object with retry strategy for robust API calls.
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = create_session()


def fetch_page(category, page, page_size):
    """
    Fetches a page of products from OpenFoodFacts.

    Parameters
    ----------
    category : str
        Product category to query.
    page : int
        Page index (starting at 1).
    page_size : int
        Number of products per page.

    Returns
    -------
    list
        List of product dictionaries.
    """
    params = {
        "categories_tags": category,
        "page": page,
        "page_size": page_size,
        "fields": PRODUCT_FIELDS,
    }

    try:
        response = SESSION.get(
            BASE_URL,
            params=params,
            headers=HEADERS,
            timeout=(10, 30)
        )
        response.raise_for_status()
        if "application/json" not in response.headers.get("content-type", ""):
            raise RuntimeError(
                f"API returned non-JSON (likely down). "
                f"status={response.status_code}, "
                f"content-type={response.headers.get('content-type')}"
            )
        return response.json().get("products", [])
    except Exception as error:
        print(f"⚠ Erreur API sur la page {page} :", error, flush=True)
        raise


def get_best_image(product):
    """
    Selects the best available image URL for a product.

    Parameters
    ----------
    product : dict
        Product metadata from OpenFoodFacts.

    Returns
    -------
    str or None
        URL of the best available image.
    """
    return (
        product.get("image_url")
        or product.get("image_front_url")
        or product.get("image_small_url")
        or product.get("image_thumb_url")
    )


def is_valid_product(product):
    """
    Checks whether a product contains the required fields.

    Parameters
    ----------
    product : dict
        Product metadata.

    Returns
    -------
    bool
        True if product is valid and has an image.
    """
    if not (product.get("code") or product.get("_id")):
        return False
    if not product.get("product_name"):
        return False
    if not product.get("categories_tags"):
        return False
    return bool(get_best_image(product))


def extract_product_info(product):
    """
    Extracts relevant fields from a product.

    Parameters
    ----------
    product : dict
        Product metadata.

    Returns
    -------
    list
        Extracted fields: id, name, categories, ingredients, image_url.
    """
    return [
        product.get("code") or product.get("_id"),
        product.get("product_name"),
        ", ".join(product.get("categories_tags", [])),
        product.get("ingredients_text", ""),
        get_best_image(product)
    ]


def save_to_csv(filename, rows):
    """
    Saves product rows to a CSV file.

    Parameters
    ----------
    filename : str
        Output CSV filename.
    rows : list of list
        Product rows to write.
    """
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["foodId", "label", "category", "foodContentsLabel", "image"]
        )
        writer.writerows(rows)


def download_image(image_url, image_id, folder="images"):
    """
    Downloads an image from a URL and saves it locally.

    Parameters
    ----------
    image_url : str
        URL of the image.
    image_id : str
        Unique product ID used as filename.
    folder : str, optional
        Output folder for images.
    """
    os.makedirs(folder, exist_ok=True)

    ext = image_url.split(".")[-1].split("?")[0]
    filename = os.path.join(folder, f"{image_id}.{ext}")

    if os.path.exists(filename):
        return

    try:
        response = SESSION.get(
            image_url,
            headers=HEADERS,
            timeout=(10, 30)
        )
        response.raise_for_status()

        with open(filename, "wb") as f:
            f.write(response.content)

    except Exception as error:
        print(f"⚠ Impossible de télécharger l'image {image_url} :", error)


def scrape_category(category):
    """
    Scrapes one category: downloads images into data/raw/images/<category>/
    and writes a per-category metadata CSV.

    Parameters
    ----------
    category : str
        OpenFoodFacts category slug (e.g. "sugar", "chocolates").
    """
    category_images_dir = os.path.join(IMAGES_ROOT, category)
    os.makedirs(category_images_dir, exist_ok=True)

    valid_products = []
    page = 1

    while len(valid_products) < TARGET_COUNT and page <= MAX_PAGES:
        print(f"[{category}] → page {page}…")

        products = fetch_page(category, page, PAGE_SIZE)
        if not products:
            print(f"[{category}] Aucun produit trouvé sur cette page.")
            break

        for product in products:
            if is_valid_product(product):
                info = extract_product_info(product)
                valid_products.append(info)

                image_url = info[-1]
                image_id = info[0]
                download_image(
                    image_url, image_id, folder=category_images_dir
                )

            if len(valid_products) == TARGET_COUNT:
                break

        page += 1
        time.sleep(0.3)

    output_file = os.path.join(
        OUTPUT_ROOT, f"metadata_{category}_{len(valid_products)}.csv"
    )
    save_to_csv(output_file, valid_products)

    print(
        f"[{category}] ✔ {output_file} créé. "
        f"Produits valides : {len(valid_products)}"
    )


def main():
    """
    Scrapes every category in CATEGORIES and organizes outputs as:
        data/raw/images/<category>/<id>.jpg
        data/raw/metadata_<category>_<n>.csv

    A failure on one category (e.g. transient API rate-limit) does not
    abort the whole run — the next category is attempted.
    """
    os.makedirs(IMAGES_ROOT, exist_ok=True)
    for category in CATEGORIES:
        try:
            scrape_category(category)
        except Exception as error:
            print(
                f"[{category}] ✗ aborted: {error}. Skipping to next.",
                flush=True,
            )


if __name__ == "__main__":
    main()
