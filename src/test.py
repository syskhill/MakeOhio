from PIL import Image
try:
    img = Image.open('src\\samC.jpg')
    img.show()
except Exception as e:
    print("Error opening image:", e)