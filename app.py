import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd

# Hardcoded answer keys from provided data (questions 1-100, options as lists for multi-correct)
key_A = {
    1: ['a'], 2: ['c'], 3: ['c'], 4: ['c'], 5: ['c'],
    6: ['a'], 7: ['c'], 8: ['c'], 9: ['b'], 10: ['c'],
    11: ['a'], 12: ['a'], 13: ['d'], 14: ['a'], 15: ['b'],
    16: ['a', 'b', 'c', 'd'], 17: ['c'], 18: ['d'], 19: ['a'], 20: ['b'],
    21: ['a'], 22: ['d'], 23: ['b'], 24: ['a'], 25: ['c'],
    26: ['b'], 27: ['a'], 28: ['b'], 29: ['d'], 30: ['c'],
    31: ['c'], 32: ['a'], 33: ['b'], 34: ['c'], 35: ['a'],
    36: ['b'], 37: ['d'], 38: ['b'], 39: ['a'], 40: ['b'],
    41: ['c'], 42: ['c'], 43: ['c'], 44: ['b'], 45: ['b'],
    46: ['a'], 47: ['c'], 48: ['b'], 49: ['d'], 50: ['a'],
    51: ['c'], 52: ['b'], 53: ['c'], 54: ['c'], 55: ['a'],
    56: ['b'], 57: ['b'], 58: ['a'], 59: ['a', 'b'], 60: ['b'],
    61: ['b'], 62: ['c'], 63: ['a'], 64: ['b'], 65: ['c'],
    66: ['b'], 67: ['b'], 68: ['c'], 69: ['c'], 70: ['b'],
    71: ['b'], 72: ['b'], 73: ['d'], 74: ['b'], 75: ['a'],
    76: ['b'], 77: ['b'], 78: ['b'], 79: ['b'], 80: ['b'],
    81: ['a'], 82: ['b'], 83: ['c'], 84: ['b'], 85: ['c'],
    86: ['b'], 87: ['b'], 88: ['b'], 89: ['a'], 90: ['b'],
    91: ['c'], 92: ['b'], 93: ['c'], 94: ['b'], 95: ['b'],
    96: ['b'], 97: ['c'], 98: ['a'], 99: ['b'], 100: ['c']
}

key_B = {
    1: ['a'], 2: ['b'], 3: ['d'], 4: ['b'], 5: ['b'],
    6: ['d'], 7: ['c'], 8: ['c'], 9: ['a'], 10: ['c'],
    11: ['a'], 12: ['b'], 13: ['d'], 14: ['c'], 15: ['c'],
    16: ['a'], 17: ['c'], 18: ['b'], 19: ['d'], 20: ['c'],
    21: ['a'], 22: ['a'], 23: ['b'], 24: ['a'], 25: ['b'],
    26: ['a'], 27: ['b'], 28: ['b'], 29: ['c'], 30: ['c'],
    31: ['b'], 32: ['c'], 33: ['b'], 34: ['c'], 35: ['a'],
    36: ['a'], 37: ['a'], 38: ['b'], 39: ['b'], 40: ['a'],
    41: ['b'], 42: ['a'], 43: ['d'], 44: ['b'], 45: ['c'],
    46: ['b'], 47: ['b'], 48: ['b'], 49: ['b'], 50: ['b'],
    51: ['c'], 52: ['a'], 53: ['c'], 54: ['a'], 55: ['c'],
    56: ['c'], 57: ['b'], 58: ['a'], 59: ['b'], 60: ['c'],
    61: ['b'], 62: ['b'], 63: ['b'], 64: ['d'], 65: ['c'],
    66: ['b'], 67: ['b'], 68: ['a'], 69: ['b'], 70: ['b'],
    71: ['b'], 72: ['c'], 73: ['a'], 74: ['d'], 75: ['b'],
    76: ['b'], 77: ['d'], 78: ['a'], 79: ['b'], 80: ['a'],
    81: ['b'], 82: ['c'], 83: ['b'], 84: ['a'], 85: ['c'],
    86: ['b'], 87: ['b'], 88: ['a'], 89: ['b'], 90: ['d'],
    91: ['c'], 92: ['d'], 93: ['b'], 94: ['b'], 95: ['b'],
    96: ['c'], 97: ['c'], 98: ['b'], 99: ['b'], 100: ['c']
}

# Helper functions for perspective transform
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Main processing function
def process_omr(image, set_no):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, None, None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    docCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break
    if docCnt is None:
        return None, None, None
    paper = four_point_transform(img, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)
    if len(questionCnts) != 400:
        st.warning(f"Incorrect number of bubbles detected: {len(questionCnts)} (expected 400).")
    questionCnts = sorted(questionCnts, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    answers = {}
    bubble_thresh = 50  # Adjust based on bubble size; pixels indicating marked
    for row in range(20):
        for sec in range(5):
            q = row + 1 + sec * 20
            start_idx = row * 20 + sec * 4
            cnts_group = questionCnts[start_idx:start_idx + 4]
            bubbled = None
            max_total = -1
            for j in range(4):
                mask = np.zeros(warped.shape, dtype="uint8")
                cv2.drawContours(mask, [cnts_group[j]], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                if total > max_total:
                    max_total = total
                    bubbled = j
            if max_total > bubble_thresh:
                answers[q] = chr(ord('A') + bubbled)
            else:
                answers[q] = None
    key = key_A if set_no == 'A' else key_B
    total_score = 0
    section_scores = {'Python': 0, 'Data Analysis': 0, 'MySQL': 0, 'Power BI': 0, 'Adv Stats': 0}
    sections = ['Python', 'Data Analysis', 'MySQL', 'Power BI', 'Adv Stats']
    for q in range(1, 101):
        marked = answers.get(q)
        correct_list = key.get(q, [])
        sec_idx = (q - 1) // 20
        if marked and marked.lower() in [opt.lower() for opt in correct_list]:
            total_score += 1
            section_scores[sections[sec_idx]] += 1
            color = (0, 255, 0)  # Green for correct
        else:
            color = (0, 0, 255) if marked else (0, 0, 0)  # Red for wrong, black for none
        if marked:
            bubbled_idx = ord(marked.upper()) - ord('A')
            cv2.drawContours(paper, [questionCnts[(q-1)*4 + bubbled_idx]], -1, color, 3)
    return total_score, section_scores, paper

# Streamlit app
st.title("Automated OMR Evaluation System")
uploaded_file = st.file_uploader("Upload OMR Sheet Image", type=["jpg", "png", "jpeg"])
set_no = st.selectbox("Set No.", ["A", "B"])
student_name = st.text_input("Student Name (for record)")

if uploaded_file and set_no:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Sheet")
    if st.button("Evaluate"):
        total_score, section_scores, processed_img = process_omr(image, set_no)
        if total_score is not None:
            st.success(f"Total Score: {total_score}/100")
            st.write("Section-wise Scores:")
            df = pd.DataFrame(list(section_scores.items()), columns=["Section", "Score"])
            st.table(df)
            # Display processed image
            processed_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            st.image(processed_pil, caption="Processed Sheet (Green: Correct, Red: Wrong)")
            # Export as CSV
            results = {"Student": student_name, "Set": set_no, "Total Score": total_score, **section_scores}
            df_export = pd.DataFrame([results])
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "results.csv", "text/csv")
        else:
            st.error("Failed to detect sheet outline. Try a clearer image.")