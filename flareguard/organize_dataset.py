import os
import shutil
import xml.etree.ElementTree as ET

image_folder = "Datacluster Fire and Smoke Sample/Datacluster Fire and Smoke Sample"
annotation_folder = "Annotations/Annotations"
dataset_fire = "dataset/fire"
dataset_smoke = "dataset/smoke"

os.makedirs(dataset_fire, exist_ok=True)
os.makedirs(dataset_smoke, exist_ok=True)

for xml_file in os.listdir(annotation_folder):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(annotation_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_tag = root.find("filename")
        if filename_tag is None:
            continue

        filename = filename_tag.text
        image_path = os.path.join(image_folder, filename)

        # Loop through objects safely
        for obj in root.findall("object"):
            name_tag = obj.find("name")
            if name_tag is None:
                continue

            label = name_tag.text.lower()

            if os.path.exists(image_path):
                if label == "fire":
                    shutil.copy(image_path, dataset_fire)
                elif label == "smoke":
                    shutil.copy(image_path, dataset_smoke)

print("Dataset organized successfully!")
label = name_tag.text.lower()
print(label)
