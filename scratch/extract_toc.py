import PyPDF2

def extract_toc():
    reader = PyPDF2.PdfReader(r"d:\Adaptive-Threat-Detection\Paper\FINAL_YEAR_THESIS_Template_CSE400_Fall_2024_ONWARDS Report_signed.pdf")
    print("Total pages:", len(reader.pages))
    
    # Let's extract first 15 pages to find the table of contents
    for i in range(5, 12):
        print(f"--- Page {i+1} ---")
        print(reader.pages[i].extract_text()[:1000])

if __name__ == "__main__":
    extract_toc()
