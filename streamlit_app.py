import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# 이미지 업로드
def upload_image():
    uploaded_file = st.file_uploader("이미지 업로드", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)  # 여기에서 원본 이미지를 표시합니다.
        return image, uploaded_file
    return None, None

# 얼굴 탐지
# def detect_faces(image):
#     # Haar Cascade 분류기
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     # OpenCV로 이미지 읽기
#     img = np.array(image)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # 얼굴 탐지
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     return faces, img

# 얼굴 탐지 함수
def detect_faces(image):
    # Haar Cascade 분류기
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # OpenCV로 이미지 읽기
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 얼굴 탐지
    # scaleFactor와 minNeighbors를 조정하여 감도를 높입니다.
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,  # 더 작은 값을 시도해보세요
        minNeighbors=3,    # 더 작은 값으로 조정
        minSize=(30, 30)
    )
    return faces, img

# 얼굴 중심으로 크롭 함수
def crop_face(image, face):
    x, y, w, h = face
    center_x = x + w // 2
    center_y = y + h // 2
    # 상하좌우 중 가장 짧은 면을 기준으로 4:3 비율 계산
    short_side = min(image.shape[0], image.shape[1], key=lambda x: x / 4 * 3)
    new_h = short_side
    new_w = int(new_h * 4 / 3)
    # 크롭 범위 계산
    start_x = max(center_x - new_w // 2, 0)
    end_x = min(center_x + new_w // 2, image.shape[1])
    start_y = max(center_y - new_h // 2, 0)
    end_y = min(center_y + new_h // 2, image.shape[0])
    # 이미지 크롭
    cropped_img = image[start_y:end_y, start_x:end_x]
    return cropped_img

def main():
    # Streamlit 페이지 설정
    st.set_page_config(page_title="Face Detection and Crop", page_icon="https://simpleicon.com/wp-content/uploads/crop.png")

    # featured image
    st.image("https://simpleicon.com/wp-content/uploads/crop.png", width=200)
    st.title("얼굴 탐지 및 이미지 크롭")
    st.write("이미지 파일을 업로드하고, 탐지된 얼굴 중 하나를 선택하여 4:3 비율로 크롭합니다.")

    image, uploaded_file = upload_image()  # 업로드된 이미지와 파일 객체 받기

    if image is not None:
        # 세션 상태에 얼굴 정보 저장
        if 'faces' not in st.session_state:
            st.session_state.faces = []

        # 얼굴 탐지 및 세션 상태 업데이트
        faces, img = detect_faces(image)
        st.session_state.faces = faces

        # 썸네일 및 라디오 버튼 표시
        if len(faces) == 0:
            st.write("얼굴이 감지되지 않았습니다.")
        else:
            # 얼굴 썸네일 표시
            for i, face in enumerate(faces, start=1):
                x, y, w, h = face
                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100))
                st.image(face_img, caption=f'Face {i}', width=100)

            # 라디오 버튼으로 얼굴 선택
            selected_face_index = st.radio("Select a face to crop", range(1, len(st.session_state.faces) + 1)) - 1

            # '자르기' 버튼
            if st.button("자르기"):
                cropped_image = crop_face(img, st.session_state.faces[selected_face_index])
                st.image(cropped_image, caption='Cropped Image', use_column_width=True)

                # 크롭된 이미지 다운로드 버튼
                buf = io.BytesIO()
                Image.fromarray(cropped_image).save(buf, format="PNG")
                byte_im = buf.getvalue()

                # 원본 이미지명 추출 및 다운로드 파일명 설정
                original_file_name, file_extension = os.path.splitext(uploaded_file.name)
                cropped_file_name = f"{original_file_name}_cropped.png"

                st.download_button(
                    label="내려받기",
                    data=byte_im,
                    file_name=cropped_file_name,
                    mime="image/png"
                )

if __name__ == "__main__":
    main()

