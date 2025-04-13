from PIL import Image

img = Image.open(r"C:\Users\Administrator\Desktop\d1.jpg")
img_resized = img.resize((32, 32))
img_resized.show()
img_resized.save(r"C:\Users\Administrator\Desktop\testd1.jpg")
8