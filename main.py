import face_recognition
from PIL import Image, ImageDraw


def face_rec():
    face_img = face_recognition.load_image_file("")
    face_location = face_recognition.face_locations(face_img)

    print(face_location)
    print(f"Found {len(face_location)} face(s) in this image.")

    pil_img = Image.fromarray(face_img)
    draw = ImageDraw.Draw(pil_img)

    for (top, right, bottom, left) in face_location:
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw
    pil_img.save("img/pil_img.jpg")


def extracting_faces(img_path):
    count = 0
    faces = face_recognition.load_image_file(img_path)
    faces_location = face_recognition.face_locations(faces)

    for face_location in faces_location:
        top, right, bottom, left = face_location

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f"img/{count}_face_img.jpg")
        count += 1

    return f"Found {count} face(s) in this image."


def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]

    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)
    print(result)


def main():
    # face_rec()
    # print(extracting_faces(""))
    # compare_faces("", "")


if __name__ == '__main__':
    main()
