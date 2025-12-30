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
        street_fov=90,
        sleep_time=0.2
    ):
        self.api_key = api_key
        self.satellite_zoom = satellite_zoom
        self.image_size = image_size
        self.street_fov = street_fov
        self.sleep_time = sleep_time


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

    def _street_view_url(self, lat, lon):
        return (
            "https://maps.googleapis.com/maps/api/streetview"
            f"?size={self.image_size}x{self.image_size}"
            f"&location={lat},{lon}"
            f"&fov={self.street_fov}"
            f"&pitch=0"
            f"&key={self.api_key}"
        )


    def _street_view_available(self, lat, lon):
        url = (
            "https://maps.googleapis.com/maps/api/streetview/metadata"
            f"?location={lat},{lon}"
            f"&key={self.api_key}"
        )

        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return False

        data = response.json()
        return data.get("status") == "OK"


    def _download_image(self, url, save_path, is_satellite=False, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)

                if response.status_code != 200:
                    return False

              
                if not is_satellite and len(response.content) < 1000:
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



    def process_dataframe(self, df, output_root):
        """
        df must contain: id, lat, long
        output_root:
        Data/train_images_CNN or Data/test_images_CNN
        """

        sat_dir = os.path.join(output_root, "satellite")
        street_dir = os.path.join(output_root, "street")

        os.makedirs(sat_dir, exist_ok=True)
        os.makedirs(street_dir, exist_ok=True)

      
        missing_sat_path = os.path.join(output_root, "missing_satellite_ids.csv")
        missing_street_path = os.path.join(output_root, "missing_street_ids.csv")

        missing_satellite_ids = set()
        missing_street_ids = set()

        if os.path.exists(missing_sat_path):
            missing_satellite_ids = set(
                pd.read_csv(missing_sat_path)["id"].astype(int)
            )

        if os.path.exists(missing_street_path):
            missing_street_ids = set(
                pd.read_csv(missing_street_path)["id"].astype(int)
            )

        newly_missing_satellite = []
        newly_missing_street = []

    
        for _, row in df.iterrows():
            image_id = int(float(row["id"]))
            lat = row["lat"]
            lon = row["long"]

            sat_path = os.path.join(sat_dir, f"{image_id}.jpg")
            street_path = os.path.join(street_dir, f"{image_id}.jpg")

            
            if image_id not in missing_satellite_ids and not os.path.exists(sat_path):
                sat_url = self._satellite_url(lat, lon)
                success = self._download_image(
    sat_url, sat_path, is_satellite=True
)


                if not success:
                    print(f"[Satellite missing] ID {image_id}")
                    newly_missing_satellite.append(image_id)

            if image_id not in missing_street_ids and not os.path.exists(street_path):
                if self._street_view_available(lat, lon):
                    street_url = self._street_view_url(lat, lon)
                    success = self._download_image(
    street_url, street_path, is_satellite=False
)

                else:
                    success = False

                if not success:
                    print(f"[Street View missing] ID {image_id}")
                    newly_missing_street.append(image_id)

 
        if newly_missing_satellite:
            all_missing_sat = missing_satellite_ids.union(newly_missing_satellite)
            pd.DataFrame({"id": sorted(all_missing_sat)}).to_csv(
                missing_sat_path, index=False
            )

        if newly_missing_street:
            all_missing_street = missing_street_ids.union(newly_missing_street)
            pd.DataFrame({"id": sorted(all_missing_street)}).to_csv(
                missing_street_path, index=False
            )

        print(
            f"Resume-safe run complete | "
            f"New satellite missing: {len(newly_missing_satellite)} | "
            f"New street missing: {len(newly_missing_street)}"
        )





import pandas as pd

train_df = pd.read_csv("Data/lat_long/image_metadata_train.csv")

main_test=pd.read_csv("Data/lat_long/image_metadata_main_test.csv")

extractor = GoogleImageExtractor(api_key=GOOGLE_API_KEY)

extractor.process_dataframe(train_df, "Data/train_images_CNN")

extractor.process_dataframe(main_test, "Data/main_test_images_CNN")
