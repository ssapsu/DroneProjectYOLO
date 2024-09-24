from ultralytics import YOLO
import cv2
import os

# 모델 로드
model = YOLO('./runs/detect/train4/weights/best.pt')

# 테스트할 이미지 경로
image_path = './Parcel-Collection-1/test/images'

for image_file in os.listdir(image_path):
    # 이미지 로드
    image = cv2.imread(os.path.join(image_path, image_file))

    # 객체 감지 수행
    results = model(image)

    # 결과 시각화
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            print(cls)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'Parcel {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 저장
    output_path = './{image_file}.jpg'.format(image_file=image_file)
    cv2.imwrite(output_path, image)

    print(f'결과 이미지가 {output_path}에 저장되었습니다.')
