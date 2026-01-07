import os
import time
import requests
import pandas as pd

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables")


class GoogleImageExtractor:
    def __init__(
        self,
        api_key,
        satellite_zoom=18,
        image_size=224,
        sleep_time=0.2
    ):
        self.api_key = api_key
        self.satellite_zoom = satellite_zoom
        self.image_size = image_size
        self.sleep_time = sleep_time

    # --------------------------------------------------
    # Satellite URL
    # --------------------------------------------------
    def _satellite_url(self, lat, lon):
        return (
            "https://maps.googleapis.com/maps/api/staticmap"
            f"?center={lat},{lon}"
            f"&zoom={self.satellite_zoom}"
            f"&size={self.image_size}x{self.image_size}"
            f"&scale=2"
            f"&maptype=satellite"
            f"&key={self.api_key}"
        )

    # --------------------------------------------------
    # Download helper (retry-safe)
    # --------------------------------------------------
    def _download_image(self, url, save_path, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)

                if response.status_code != 200:
                    return False

                with open(save_path, "wb") as f:
                    f.write(response.content)

                time.sleep(self.sleep_time)
                return True

            except requests.exceptions.ReadTimeout:
                time.sleep((attempt + 1) * 2)

            except requests.exceptions.RequestException:
                return False

        return False

    # --------------------------------------------------
    # Main processing (RESUME-SAFE)
    # --------------------------------------------------
    def process_dataframe(self, df, output_root):
        """
        df must contain: id, lat, long
        """

        sat_dir = os.path.join(output_root, "satellite")
        os.makedirs(sat_dir, exist_ok=True)

        missing_sat_path = os.path.join(output_root, "missing_satellite_ids.csv")

        # --------------------------------------------------
        # Load known-missing IDs (AS STRINGS)
        # --------------------------------------------------
        missing_satellite_ids = set()
        if os.path.exists(missing_sat_path):
            missing_satellite_ids = set(
                pd.read_csv(missing_sat_path)["id"].astype(str).str.strip()
            )

        newly_missing = []

        # --------------------------------------------------
        # Iterate rows
        # --------------------------------------------------
        for _, row in df.iterrows():
            image_id = str(row["id"]).strip()   # ✅ FIX
            lat = row["lat"]
            lon = row["long"]

            sat_path = os.path.join(sat_dir, f"{image_id}.jpg")

            # Skip if already known missing or already downloaded
            if image_id in missing_satellite_ids or os.path.exists(sat_path):
                continue

            sat_url = self._satellite_url(lat, lon)
            success = self._download_image(sat_url, sat_path)

            if not success:
                print(f"[Satellite missing] ID {image_id}")
                newly_missing.append(image_id)

        # --------------------------------------------------
        # Persist missing IDs (append-safe)
        # --------------------------------------------------
        if newly_missing:
            all_missing = missing_satellite_ids.union(newly_missing)
            pd.DataFrame({"id": sorted(all_missing)}).to_csv(
                missing_sat_path, index=False
            )

        print(
            f"Resume-safe run complete | "
            f"New satellite missing: {len(newly_missing)}"
        )


import pandas as pd

train_df = pd.read_csv("Data/lat_long/image_metadata_train.csv")

main_test=pd.read_csv("Data/lat_long/image_metadata_main_test.csv")

extractor = GoogleImageExtractor(api_key=GOOGLE_API_KEY)

extractor.process_dataframe(train_df, "Data/train_images_CNN")

extractor.process_dataframe(main_test, "Data/main_test_images_CNN")
