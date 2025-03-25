import cv2

from ultralytics import YOLOv10

weights = './runs/detect/train/weights/best.pt'
model = YOLOv10(weights)
img = 'test3.jpeg'
results = model.predict(img)

boxes = []
for r in results:
    print(r.boxes.xywh)
    # r.show()
    boxes.append(r.boxes.xywh)

image = cv2.imread(img)
for box in boxes:
    x, y, w, h = box[0]
    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow('image', image)
    cropped_image = image[y1:y2, x1:x2]
    cv2.imshow('Cropped Image', cv2.resize(cropped_image, (640, 640)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

