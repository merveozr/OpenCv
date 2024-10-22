import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
font1 = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread("C:\\Users\\Merve\\Desktop\\openCv\\shape_detection\\polygons.png")
img1=cv2.imread("C:\\Users\\Merve\\Desktop\\openCv\\shape_detection\\shape.png")
img2 = cv2.imread("C:\\Users\\Merve\\Desktop\\openCv\\shape_detection\\shape1.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
_, threshold1 = cv2.threshold(gray1, 240, 255, cv2.THRESH_BINARY)
_, threshold2 = cv2.threshold(gray2, 240, 255, cv2.THRESH_BINARY)

a, b, c, d, e = 0, 0, 0, 0, 0

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours1, _ = cv2.findContours(threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(threshold2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

kernel = np.ones((5, 5), np.uint8)
img_dilation = cv2.dilate(img2, kernel, iterations=2)

# Genişletilmiş renk algılama, pembe eklendi
def get_color_name(B, G, R):
    if R > 150 and G < 100 and B < 100:
        return "Red"
    elif R < 100 and G > 150 and B < 100:
        return "Green"
    elif R < 100 and G < 100 and B > 150:
        return "Blue"
    elif R > 150 and G > 150 and B < 100:
        return "Yellow"
    elif R > 150 and G < 100 and B > 150:
        return "Magenta"
    elif R > 150 and G < 150 and B > 150:
        return "Pink"  # Pembe eklendi
    elif R < 100 and G > 150 and B > 150:
        return "Cyan"
    elif R > 200 and G > 200 and B > 200:
        return "White"
    elif R < 50 and G < 50 and B < 50:
        return "Black"
    else:
        return "Undefined Color"

def process_contours(image, contours):
    global a, b, c, d, e
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        cv2.drawContours(image, [approx], 0, (0, 0, 0), 5)

        x = approx.ravel()[0]
        y = approx.ravel()[1]

        # Şeklin içindeki ortalama rengi hesapla
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(image, mask=mask)

        # Renk ismini belirle
        color_name = get_color_name(mean_val[0], mean_val[1], mean_val[2])

        if len(approx) == 3:
            cv2.putText(image, f"Triangle - {color_name}", (x, y), font1, 1, (0, 0, 0))
            a += 1
        elif len(approx) == 4:
            cv2.putText(image, f"Rectangle - {color_name}", (x, y), font1, 1, (0, 0, 0))
            b += 1
        elif len(approx) == 5:
            cv2.putText(image, f"Pentagon - {color_name}", (x, y), font, 1, (0, 0, 0))
            c += 1
        elif len(approx) == 6:
            cv2.putText(image, f"Hexagon - {color_name}", (x, y), font, 1, (0, 0, 0))
            d += 1
        else:
            cv2.putText(image, f"Ellipse - {color_name}", (x, y), font, 1, (0, 0, 0))
            e += 1

process_contours(img, contours)
process_contours(img1, contours1)
process_contours(img_dilation, contours2)

print(f"triangle={a}, rectangle={b}, pentagon={c}, hexagon={d}, ellipse={e}")

cv2.imshow("IMG", img)
cv2.imshow("IMG1", img1)
cv2.imshow("IMG2", img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

