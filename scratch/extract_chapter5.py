import PyPDF2

def extract_chapter5():
    reader = PyPDF2.PdfReader(r"d:\Adaptive-Threat-Detection\Paper\FINAL_YEAR_THESIS_Template_CSE400_Fall_2024_ONWARDS Report_signed.pdf")
    with open(r"d:\Adaptive-Threat-Detection\scratch\example_chapter5.txt", "w", encoding="utf-8") as f:
        # Chapter 5 starts on page 47 (index 46) to 58 (index 57)
        for i in range(46, 58):
            f.write(f"================ PAGE {i+1} ================\n")
            f.write(reader.pages[i].extract_text() + "\n")

if __name__ == "__main__":
    extract_chapter5()
