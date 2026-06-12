import zipfile

def list_zip():
    with zipfile.ZipFile(r"d:\Adaptive-Threat-Detection\thesis_materials\Thesis.zip", "r") as z:
        for name in z.namelist()[:30]:
            print(name)

if __name__ == "__main__":
    list_zip()
