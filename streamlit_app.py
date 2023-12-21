import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from retinaface import RetinaFace  # Importing RetinaFace

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

# 얼굴 탐지 함수 - Now using RetinaFace
def detect_faces(image):
    img = image.convert('RGB')
    img = np.array(img)

    # RetinaFace를 사용한 얼굴 탐지
    faces = RetinaFace.detect_faces(img_path=img)
    if faces is None:
        return [], img

    detected_faces = []
    for face in faces.values():
        facial_area = face['facial_area']
        x, y, w, h = facial_area
        detected_faces.append((x, y, w-x, h-y))  # Converting to (x, y, w, h) format
    return detected_faces, img

# 얼굴을 중심으로 원본 크기를 넘지 않도록 4:3 비율로 크롭하는 로직
def crop_face_original(image, face):
    x, y, w, h = face
    center_x = x + w // 2
    center_y = y + h // 2

    img_height, img_width = image.shape[:2]
    aspect_ratio = 4 / 3

    # 가로, 세로 중 어느 쪽을 기준으로 크롭할지 결정
    if img_width / img_height >= aspect_ratio:
        # 이미지가 가로로 길거나 정사각형일 때
        crop_width = int(img_height * aspect_ratio)
        crop_height = img_height
    else:
        # 이미지가 세로로 길 때
        crop_width = img_width
        crop_height = int(img_width / aspect_ratio)

    print(f"가로: {crop_width}, 세로: {crop_height}")

    # 크롭 영역이 이미지 경계를 넘어가지 않도록 조정
    start_x = max(center_x - crop_width // 2, 0)
    end_x = start_x + crop_width
    if end_x > img_width:
        end_x = img_width
        start_x = end_x - crop_width

    start_y = max(center_y - crop_height // 2, 0)
    end_y = start_y + crop_height
    if end_y > img_height:
        end_y = img_height
        start_y = end_y - crop_height
    
    # 이미지 크롭
    cropped_img = image[start_y:end_y, start_x:end_x]
    return cropped_img

# 얼굴을 기준으로 최솟값 범위 안에서 크롭하는 로직
def crop_face_minimum(image, face):
    x, y, w, h = face
    center_x = x + w // 2
    center_y = y + h // 2

    img_height, img_width = image.shape[:2]
    aspect_ratio = 4 / 3

    # 가로 또는 세로 길이가 300px 미만일 때 4:3 비율로 크롭
    if img_width < 300 or img_height < 300:
        # 가장 짧은 쪽을 기준으로 크롭 비율 설정
        base = min(img_width, img_height)
        if base == img_width:  # 가로가 더 짧은 경우
            crop_width = base
            crop_height = int(crop_width / aspect_ratio)
        else:  # 세로가 더 짧은 경우
            crop_height = base
            crop_width = int(crop_height * aspect_ratio)
        
        # 선택된 얼굴이 크롭 영역 내에 있는지 확인하고 조정
        start_x = max(center_x - crop_width // 2, 0)
        end_x = min(start_x + crop_width, img_width)
        start_y = max(center_y - crop_height // 2, 0)
        end_y = min(start_y + crop_height, img_height)
        
        # 크롭 영역이 이미지 경계를 넘지 않도록 재조정
        if end_x - start_x < crop_width:
            start_x = end_x - crop_width
        if end_y - start_y < crop_height:
            start_y = end_y - crop_height
    else:
        # 원본 이미지가 큰 경우 얼굴 크기를 기준으로 크롭 영역 계산
        crop_width = max(300, w * 1.5)
        crop_height = int(crop_width / aspect_ratio)
        
        print(f"crop_width: {crop_width}, crop_height: {crop_height}")

        # 크롭 영역이 선택된 얼굴을 포함하도록 조정
        start_x = max(center_x - crop_width // 2, 0)
        end_x = min(start_x + crop_width, img_width)
        start_y = max(center_y - crop_height // 2, 0)
        end_y = min(start_y + crop_height, img_height)

    # 이미지 크롭
    cropped_img = image[start_y:end_y, start_x:end_x]
    return cropped_img

def main():
    # Streamlit 페이지 설정
    st.set_page_config(page_title="Face Detection and Crop", page_icon="https://simpleicon.com/wp-content/uploads/crop.png")

    # featured image
    st.image("https://simpleicon.com/wp-content/uploads/crop.png", width=200)
    st.title("얼굴 탐지 및 이미지 크롭")
    st.write("이미지 파일을 업로드하고, 탐지된 얼굴 중 하나를 선택해 4:3 비율로 크롭합니다.")

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
            
            # 크롭 방식 선택
            crop_method = st.radio("Choose the crop method", ["얼굴만 자르기", "주변을 포함해 자르기"])

            # '자르기' 버튼
            if st.button("자르기"):
                if crop_method == "얼굴만 자르기":
                    cropped_image = crop_face_minimum(img, faces[selected_face_index])
                else:
                    cropped_image = crop_face_original(img, faces[selected_face_index])

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
