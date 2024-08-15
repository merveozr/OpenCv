import cv2
import numpy as np
import os

image_dir = '/home/merve/Masaüstü/random'  
output_file_path = '/home/merve/Masaüstü/results.txt' 

# Dosyaları alın
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Sonuçları kaydedecek dosyayı açın
with open(output_file_path, 'w') as output_file:
    for image_file in image_files:
        # Resmi yükleyin
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        
        # Görüntüyü gri tonlamaya çevirin
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültüyü azaltmak için Gaussian Blur uygulayın
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Fonksiyon, Hough dönüşümünün bir modifikasyonunu kullanarak gri tonlamalı bir görüntüde daireler bulur.
        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1,
            minDist=20, 
            param1=50, 
            param2=1,  # Bu değeri artırmak çember tespitini etkileyebilir
            minRadius=1, 
            maxRadius=10
        )
        
        output_file.write(f"Image: {image_file}\n")
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, radius = circle
                # Çember çizimi
                cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
                #output_file.write(f"x: {x}, y: {y}, radius: {radius}\n")
                print(f"x: {x}, y: {y}, radius: {radius}")
        else:
            output_file.write("No circles detected.\n")
        
        # Görüntüyü göster
        cv2.imshow("img", image)  
        cv2.waitKey(0)  # Bir tuşa basmayı bekler
        cv2.destroyAllWindows()  # Pencereyi kapatır
        
        # Görseller arasında bir boş satır ekleyin
        output_file.write("\n")

print("Daire tespiti tamamlandı ve sonuçlar kaydedildi.")

